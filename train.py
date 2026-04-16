import os
import re
import shutil
import json
from typing import Annotated, Any, Optional
import typer
import time
import toml
from model import CEmodel
from model_wdm import CEmodelWDMDemux
from model_sdm import CEmodelSDMDemux
import orbax.checkpoint as ocp
from flax import nnx
from loguru import logger
from dataset import prepare_io_dataset, prepare_io_dataset_wdm, decode_output
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np


# 创建一个 Typer 应用程序
app = typer.Typer()


def _parse_override_value(raw_value: str) -> Any:
    key = "__override_value__"
    try:
        return toml.loads(f"{key} = {raw_value}")[key]
    except toml.TomlDecodeError:
        # 允许未加引号的简单字符串，例如 fdfd_solver
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", raw_value):
            return raw_value
        raise typer.BadParameter(
            f"无法解析 override 值: {raw_value}。"
            "请使用 TOML 字面量（如 true, 10, 1e-3, \"text\", [1,2]）。"
        )


def _split_override_path(path: str) -> list[str]:
    normalized = re.sub(r"\[(\d+)\]", r".\1", path.strip())
    tokens = [token for token in normalized.split(".") if token]
    if not tokens:
        raise typer.BadParameter(f"无效 override 路径: {path}")
    return tokens


def _set_config_value(root: Any, path_tokens: list[str], value: Any) -> None:
    node = root
    for idx, token in enumerate(path_tokens[:-1]):
        next_token = path_tokens[idx + 1]

        if isinstance(node, dict):
            if token not in node:
                node[token] = [] if next_token.isdigit() else {}
            node = node[token]
            continue

        if isinstance(node, list):
            if not token.isdigit():
                raise typer.BadParameter(
                    f"路径段 {token} 应为列表索引，完整路径: {'.'.join(path_tokens)}"
                )
            index = int(token)
            if index < 0 or index >= len(node):
                raise typer.BadParameter(
                    f"列表索引越界: {index}，完整路径: {'.'.join(path_tokens)}"
                )
            node = node[index]
            continue

        raise typer.BadParameter(f"路径无法继续深入: {'.'.join(path_tokens)}")

    leaf = path_tokens[-1]
    if isinstance(node, dict):
        node[leaf] = value
        return

    if isinstance(node, list):
        if not leaf.isdigit():
            raise typer.BadParameter(
                f"路径段 {leaf} 应为列表索引，完整路径: {'.'.join(path_tokens)}"
            )
        index = int(leaf)
        if index < 0 or index >= len(node):
            raise typer.BadParameter(
                f"列表索引越界: {index}，完整路径: {'.'.join(path_tokens)}"
            )
        node[index] = value
        return

    raise typer.BadParameter(f"路径无法赋值: {'.'.join(path_tokens)}")


def apply_overrides(config: dict, overrides: list[str]) -> list[str]:
    applied = []
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(
                f"override 格式错误: {item}。应为 key=value，例如 simulation.backend=\"fdfd_solver\""
            )

        path, raw_value = item.split("=", 1)
        path_tokens = _split_override_path(path)
        value = _parse_override_value(raw_value.strip())
        _set_config_value(config, path_tokens, value)
        applied.append(f"{'.'.join(path_tokens)}={repr(value)}")

    return applied


def _get_task_type(config: dict) -> str:
    task_cfg = config.get("task", {})
    if isinstance(task_cfg, str):
        return task_cfg
    if isinstance(task_cfg, dict):
        return str(task_cfg.get("type", "standard"))
    return "standard"


def _find_latest_epoch(model_path: str) -> Optional[str]:
    if not os.path.isdir(model_path):
        return None

    epoch_list = []
    for name in os.listdir(model_path):
        if name.startswith("epoch_"):
            try:
                epoch_list.append((int(name.split("_", 1)[1]), name))
            except ValueError:
                continue

    if not epoch_list:
        return None

    _, epoch_dir = max(epoch_list, key=lambda x: x[0])
    return epoch_dir


def _save_bool(save_load_config: dict) -> bool:
    if "save" in save_load_config:
        return bool(save_load_config["save"])
    return bool(save_load_config.get("save_bool", False))


def _viz_bool(save_load_config: dict) -> bool:
    if "save_pics" in save_load_config:
        return bool(save_load_config["save_pics"])
    return bool(save_load_config.get("save_pics_bool", False))


def _train_standard(
    config: dict,
    pwd: str,
    run_dir: str,
    ckptr: ocp.AsyncCheckpointer,
):
    train_config = config["train"]
    save_load_config = config["save_load"]
    enable_jit = bool(train_config.get("enable_jit", False))

    if "load_path" in save_load_config:
        logger.info(f"加载模型和优化器从: {save_load_config['load_path']}")
        model = CEmodel(config)
        graphdef, state = nnx.split(model)
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/model", state)
        model = nnx.merge(graphdef, state)

        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))
        graphdef, state = nnx.split(optimizer)
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/optimizer", state)
        optimizer = nnx.merge(graphdef, state)
        optimizer.model = model
    else:
        logger.info("初始化模型从零开始")
        model = CEmodel(config)
        logger.info(f"初始化优化器以步长: {train_config['step_size']}")
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

    _, masks, expected_output, raw_i, raw_o = prepare_io_dataset(config["dataset"])

    def loss_fn(model, masks):
        pred = model(masks)
        return jnp.mean(jnp.sum((pred - expected_output) ** 2, axis=1)), pred

    def train_step_impl(model, optimizer, masks):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (mse, pred), grads = grad_fn(model, masks)
        pred_01 = jnp.where(pred > 0.5, 1.0, 0.0)
        pred_label = jax.vmap(decode_output)(pred_01)
        accu = jnp.mean(pred_label == raw_o.astype(int))
        optimizer.update(grads)
        return mse, accu

    train_step = nnx.jit(train_step_impl) if enable_jit else train_step_impl
    save_enabled = _save_bool(save_load_config)
    save_interval = int(save_load_config.get("save_interval", 1))
    viz_interval = save_load_config.get("viz_interval")

    train_total_start = time.time()

    for epoch in range(train_config["num_epochs"]):
        t_start = time.time()
        try:
            mse, accu = train_step(model, optimizer, masks)
        except Exception as exc:
            if enable_jit:
                logger.warning(f"JIT 执行失败，回退到非 JIT：{exc}")
                enable_jit = False
                train_step = train_step_impl
                mse, accu = train_step(model, optimizer, masks)
            else:
                raise

        t_elapsed = time.time() - t_start
        logger.info(
            f"第 {epoch} 个epoch，均方误差: {mse}, 分类准确率: {accu}, 时间: {t_elapsed}s"
        )

        if save_enabled and (epoch - 1) % save_interval == 0:
            _, state = nnx.split(model)
            ckptr.save(
                f"{pwd}/{run_dir}/models/epoch_{epoch}/model",
                args=ocp.args.PyTreeSave(state),
            )
            _, state = nnx.split(optimizer)
            ckptr.save(
                f"{pwd}/{run_dir}/models/epoch_{epoch}/optimizer",
                args=ocp.args.PyTreeSave(state),
            )

        if viz_interval is not None and (epoch - 1) % int(viz_interval) == 0:
            os.makedirs(f"{run_dir}/viz/epoch_{epoch}", exist_ok=True)
            for ndx, mask in enumerate(masks):
                fig, _ = model.viz_abs(mask)
                plt.savefig(f"{run_dir}/viz/epoch_{epoch}/{raw_i[ndx]}_{raw_o[ndx]}.png")
                plt.close(fig)

    train_total_elapsed = time.time() - train_total_start
    logger.info(f"训练总用时（epoch阶段）: {train_total_elapsed:.3f}s")


def _train_wdm(
    config: dict,
    pwd: str,
    run_dir: str,
    ckptr: ocp.AsyncCheckpointer,
):
    train_config = config["train"]
    save_load_config = config["save_load"]
    enable_jit = bool(train_config.get("enable_jit", False))

    epoch_hist: list[int] = []
    pred_hist: list[float] = []
    loss_hist: list[float] = []
    time_hist: list[float] = []
    pred_initial = 0.0

    if "load_path" in save_load_config:
        load_path = f"{pwd}/{save_load_config['load_path']}"
        model_path = f"{load_path}/models"
        latest_epoch = _find_latest_epoch(model_path)

        if os.path.isfile(f"{load_path}/train_info.json"):
            with open(f"{load_path}/train_info.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                epoch_hist = list(data.get("Epoch", []))
                pred_hist = list(data.get("Pred", []))
                loss_hist = list(data.get("Loss", []))
                time_hist = list(data.get("Time", []))

        model = CEmodelWDMDemux(config)
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

        if latest_epoch is not None:
            logger.info(f"加载模型和优化器从: {save_load_config['load_path']}/models/{latest_epoch}")

            graphdef, state = nnx.split(model)
            state = ckptr.restore(f"{model_path}/{latest_epoch}/model", state)
            model = nnx.merge(graphdef, state)

            optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))
            graphdef, state = nnx.split(optimizer)
            state = ckptr.restore(f"{model_path}/{latest_epoch}/optimizer", state)
            optimizer = nnx.merge(graphdef, state)
            optimizer.model = model
    else:
        logger.info("初始化 WDM 模型从零开始")
        model = CEmodelWDMDemux(config)
        logger.info(f"初始化优化器以步长: {train_config['step_size']}")
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

    os.makedirs(f"{run_dir}/train_info/Pred", exist_ok=True)
    os.makedirs(f"{run_dir}/train_info/Loss", exist_ok=True)

    _, masks, expected_output, _, _ = prepare_io_dataset_wdm(config["dataset"])

    logger.info(f"Epsr binarized status: {config['basic'].get('bin_bool', False)}")
    logger.info(f"Loss function type: {train_config.get('loss_type', 'MSE')}")

    patience_count = 0
    patience_max = int(train_config.get("patience_max", 20))
    patience_mul_rate = float(train_config.get("patience_mul_rate", 1.4))
    logger.info(
        f"Learning rate will be multiplied with {patience_mul_rate} if no Pred increment in {patience_max} epochs continuously."
    )

    proj_beta_stair_size = list(train_config.get("proj_beta_stair_size", []))
    proj_beta_stair_size_idx = 0

    def loss_fn_mse(model, masks):
        pred = model(masks)
        return jnp.mean((pred - 1e3) ** 2), pred

    def loss_fn_bce(model, masks):
        pred = jnp.atleast_1d(model(masks))
        tgt = expected_output.reshape((-1,))[: pred.shape[0]]
        pred = jnp.clip(pred, 1e-6, 1 - 1e-6)
        term_0 = (1 - tgt) * jnp.log(1 - pred)
        term_1 = tgt * jnp.log(pred)
        return -jnp.sum(term_0 + term_1), pred

    def loss_fn_ga(model, masks):
        pred = model(masks)
        return -jnp.mean(jnp.atleast_1d(pred)), pred

    def train_step_impl(model, optimizer, masks):
        loss_type = str(train_config.get("loss_type", "MSE")).upper()
        if loss_type == "MSE":
            grad_fn = nnx.value_and_grad(loss_fn_mse, has_aux=True)
        elif loss_type == "BCE":
            grad_fn = nnx.value_and_grad(loss_fn_bce, has_aux=True)
        elif loss_type == "GA":
            grad_fn = nnx.value_and_grad(loss_fn_ga, has_aux=True)
        else:
            raise ValueError(f"Unsupported loss_fn type: {loss_type}")

        (loss, pred), grads = grad_fn(model, masks)
        optimizer.update(grads)
        return loss, pred

    train_step = nnx.jit(train_step_impl) if enable_jit else train_step_impl
    save_enabled = _save_bool(save_load_config)
    save_pics_enabled = _viz_bool(save_load_config)
    pred_threshold = float(train_config.get("pred_threshold", 25.0))

    train_total_start = time.time()

    for epoch in range(train_config["num_epochs"]):
        t_start = time.time()
        try:
            loss, pred = train_step(model, optimizer, masks)
        except Exception as exc:
            if enable_jit:
                logger.warning(f"WDM JIT 执行失败，回退到非 JIT：{exc}")
                enable_jit = False
                train_step = train_step_impl
                loss, pred = train_step(model, optimizer, masks)
            else:
                raise
        t_elapsed = time.time() - t_start

        pred_scalar = float(np.mean(np.array(jnp.atleast_1d(pred))))
        loss_scalar = float(np.mean(np.array(jnp.atleast_1d(loss))))
        logger.info(
            f"Epoch {epoch}/{train_config['num_epochs']}, {train_config.get('loss_type', 'MSE')}: {loss_scalar}, Pred: {pred_scalar}, Time: {t_elapsed}s"
        )

        epoch_hist.append(epoch)
        pred_hist.append(pred_scalar)
        loss_hist.append(loss_scalar)
        time_hist.append(t_elapsed)

        if pred_scalar > pred_initial or epoch == train_config["num_epochs"] - 1:
            train_info = {
                "Epoch": epoch_hist,
                "Pred": pred_hist,
                "Loss": loss_hist,
                "Time": time_hist,
            }
            with open(f"{run_dir}/train_info.json", "w", encoding="utf-8") as fp:
                json.dump(train_info, fp, indent=2)

            plt.plot(pred_hist)
            plt.grid()
            plt.title(
                f"Epoch={epoch}, {train_config.get('loss_type', 'MSE')}={loss_scalar:.3e}, Pred={pred_scalar:.3e}"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Pred")
            plt.savefig(f"{run_dir}/train_info/Pred/epoch{epoch}.png")
            plt.clf()

            plt.plot(loss_hist)
            plt.grid()
            plt.title(
                f"Epoch={epoch}, {train_config.get('loss_type', 'MSE')}={loss_scalar:.3e}, Pred={pred_scalar:.3e}"
            )
            plt.xlabel("Epoch")
            plt.ylabel(train_config.get("loss_type", "MSE"))
            plt.savefig(f"{run_dir}/train_info/Loss/epoch{epoch}.png")
            plt.clf()

            pred_initial = pred_scalar

            if save_enabled:
                _, state = nnx.split(model)
                ckptr.save(
                    f"{pwd}/{run_dir}/models/epoch_{epoch}/model",
                    args=ocp.args.PyTreeSave(state),
                )
                _, state = nnx.split(optimizer)
                ckptr.save(
                    f"{pwd}/{run_dir}/models/epoch_{epoch}/optimizer",
                    args=ocp.args.PyTreeSave(state),
                )

            if save_pics_enabled:
                os.makedirs(f"{run_dir}/viz", exist_ok=True)
                for midx, mask in enumerate(masks):
                    fig, _ = model.viz_abs(mask)
                    fig.subplots_adjust(wspace=0.4)
                    plt.savefig(f"{run_dir}/viz/epoch_{epoch}_mask_{midx}.png")
                    plt.close(fig)

            patience_count = 0
        else:
            patience_count += 1

        if pred_scalar >= pred_threshold:
            if config["basic"].get("bin_bool", False):
                if model.layers[0].basic.get("proj_beta", 0.0) >= model.layers[0].basic.get(
                    "proj_beta_max", 0.0
                ):
                    break

                pred_initial = 0.0
                if train_config.get("proj_beta_stair_bool", False) and proj_beta_stair_size:
                    stair_idx = min(proj_beta_stair_size_idx, len(proj_beta_stair_size) - 1)
                    step = float(proj_beta_stair_size[stair_idx])
                    logger.info(
                        f"proj_beta {model.layers[0].basic.get('proj_beta', 0.0)} has reached Pred threshold {pred_threshold}!"
                    )
                    for i, single_layer in enumerate(model.layers):
                        single_layer.basic["proj_beta"] = single_layer.basic.get("proj_beta", 0.0) + step
                        logger.info(
                            f"proj_beta of layer{i + 1} increased {step} to {single_layer.basic['proj_beta']}"
                        )
                    proj_beta_stair_size_idx += 1
            else:
                break

    train_total_elapsed = time.time() - train_total_start
    logger.info(f"训练总用时（epoch阶段）: {train_total_elapsed:.3f}s")


def _train_sdm(
    config: dict,
    pwd: str,
    run_dir: str,
    ckptr: ocp.AsyncCheckpointer,
):
    train_config = config["train"]
    save_load_config = config["save_load"]
    enable_jit = bool(train_config.get("enable_jit", False))

    epoch_hist: list[int] = []
    pred_hist: list[float] = []
    loss_hist: list[float] = []
    time_hist: list[float] = []
    pred_initial = 0.0

    if "load_path" in save_load_config:
        load_path = f"{pwd}/{save_load_config['load_path']}"
        model_path = f"{load_path}/models"
        latest_epoch = _find_latest_epoch(model_path)

        if os.path.isfile(f"{load_path}/train_info.json"):
            with open(f"{load_path}/train_info.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                epoch_hist = list(data.get("Epoch", []))
                pred_hist = list(data.get("Pred", []))
                loss_hist = list(data.get("Loss", []))
                time_hist = list(data.get("Time", []))

        model = CEmodelSDMDemux(config)
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

        if latest_epoch is not None:
            logger.info(f"加载模型和优化器从: {save_load_config['load_path']}/models/{latest_epoch}")

            graphdef, state = nnx.split(model)
            state = ckptr.restore(f"{model_path}/{latest_epoch}/model", state)
            model = nnx.merge(graphdef, state)

            optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))
            graphdef, state = nnx.split(optimizer)
            state = ckptr.restore(f"{model_path}/{latest_epoch}/optimizer", state)
            optimizer = nnx.merge(graphdef, state)
            optimizer.model = model
    else:
        logger.info("初始化 SDM 模型从零开始")
        model = CEmodelSDMDemux(config)
        logger.info(f"初始化优化器以步长: {train_config['step_size']}")
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

    os.makedirs(f"{run_dir}/train_info/Pred", exist_ok=True)
    os.makedirs(f"{run_dir}/train_info/Loss", exist_ok=True)

    _, masks, expected_output, _, _ = prepare_io_dataset_wdm(config["dataset"])

    logger.info(f"Epsr binarized status: {config['basic'].get('bin_bool', False)}")
    logger.info(f"Loss function type: {train_config.get('loss_type', 'MSE')}")

    patience_count = 0
    patience_max = int(train_config.get("patience_max", 20))
    patience_mul_rate = float(train_config.get("patience_mul_rate", 1.4))
    logger.info(
        f"Learning rate will be multiplied with {patience_mul_rate} if no Pred increment in {patience_max} epochs continuously."
    )

    proj_beta_stair_size = list(train_config.get("proj_beta_stair_size", []))
    proj_beta_stair_size_idx = 0

    def loss_fn_mse(model, masks):
        pred = model(masks)
        return jnp.mean((pred - 1e3) ** 2), pred

    def loss_fn_bce(model, masks):
        pred = jnp.atleast_1d(model(masks))
        tgt = expected_output.reshape((-1,))[: pred.shape[0]]
        pred = jnp.clip(pred, 1e-6, 1 - 1e-6)
        term_0 = (1 - tgt) * jnp.log(1 - pred)
        term_1 = tgt * jnp.log(pred)
        return -jnp.sum(term_0 + term_1), pred

    def loss_fn_ga(model, masks):
        pred = model(masks)
        return -jnp.mean(jnp.atleast_1d(pred)), pred

    def train_step_impl(model, optimizer, masks):
        loss_type = str(train_config.get("loss_type", "MSE")).upper()
        if loss_type == "MSE":
            grad_fn = nnx.value_and_grad(loss_fn_mse, has_aux=True)
        elif loss_type == "BCE":
            grad_fn = nnx.value_and_grad(loss_fn_bce, has_aux=True)
        elif loss_type == "GA":
            grad_fn = nnx.value_and_grad(loss_fn_ga, has_aux=True)
        else:
            raise ValueError(f"Unsupported loss_fn type: {loss_type}")

        (loss, pred), grads = grad_fn(model, masks)
        optimizer.update(grads)
        return loss, pred

    train_step = nnx.jit(train_step_impl) if enable_jit else train_step_impl
    save_enabled = _save_bool(save_load_config)
    save_pics_enabled = _viz_bool(save_load_config)
    pred_threshold = float(train_config.get("pred_threshold", 100.0))

    train_total_start = time.time()

    for epoch in range(train_config["num_epochs"]):
        t_start = time.time()
        try:
            loss, pred = train_step(model, optimizer, masks)
        except Exception as exc:
            if enable_jit:
                logger.warning(f"SDM JIT 执行失败，回退到非 JIT：{exc}")
                enable_jit = False
                train_step = train_step_impl
                loss, pred = train_step(model, optimizer, masks)
            else:
                raise
        t_elapsed = time.time() - t_start

        pred_scalar = float(np.mean(np.array(jnp.atleast_1d(pred))))
        loss_scalar = float(np.mean(np.array(jnp.atleast_1d(loss))))
        logger.info(
            f"Epoch {epoch}/{train_config['num_epochs']}, {train_config.get('loss_type', 'MSE')}: {loss_scalar}, Pred: {pred_scalar}, Time: {t_elapsed}s"
        )

        epoch_hist.append(epoch)
        pred_hist.append(pred_scalar)
        loss_hist.append(loss_scalar)
        time_hist.append(t_elapsed)

        if pred_scalar > pred_initial or epoch == train_config["num_epochs"] - 1:
            train_info = {
                "Epoch": epoch_hist,
                "Pred": pred_hist,
                "Loss": loss_hist,
                "Time": time_hist,
            }
            with open(f"{run_dir}/train_info.json", "w", encoding="utf-8") as fp:
                json.dump(train_info, fp, indent=2)

            plt.plot(pred_hist)
            plt.grid()
            plt.title(
                f"Epoch={epoch}, {train_config.get('loss_type', 'MSE')}={loss_scalar:.3e}, Pred={pred_scalar:.3e}"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Pred")
            plt.savefig(f"{run_dir}/train_info/Pred/epoch{epoch}.png")
            plt.clf()

            plt.plot(loss_hist)
            plt.grid()
            plt.title(
                f"Epoch={epoch}, {train_config.get('loss_type', 'MSE')}={loss_scalar:.3e}, Pred={pred_scalar:.3e}"
            )
            plt.xlabel("Epoch")
            plt.ylabel(train_config.get("loss_type", "MSE"))
            plt.savefig(f"{run_dir}/train_info/Loss/epoch{epoch}.png")
            plt.clf()

            pred_initial = pred_scalar

            if save_enabled:
                _, state = nnx.split(model)
                ckptr.save(
                    f"{pwd}/{run_dir}/models/epoch_{epoch}/model",
                    args=ocp.args.PyTreeSave(state),
                )
                _, state = nnx.split(optimizer)
                ckptr.save(
                    f"{pwd}/{run_dir}/models/epoch_{epoch}/optimizer",
                    args=ocp.args.PyTreeSave(state),
                )

            if save_pics_enabled:
                os.makedirs(f"{run_dir}/viz", exist_ok=True)
                for midx, mask in enumerate(masks):
                    fig, _ = model.viz_abs(mask)
                    fig.subplots_adjust(wspace=0.4)
                    plt.savefig(f"{run_dir}/viz/epoch_{epoch}_mask_{midx}.png")
                    plt.close(fig)

            patience_count = 0
        else:
            patience_count += 1

        if pred_scalar >= pred_threshold:
            if config["basic"].get("bin_bool", False):
                if model.layers[0].basic.get("proj_beta", 0.0) >= model.layers[0].basic.get(
                    "proj_beta_max", 0.0
                ):
                    break

                pred_initial = 0.0
                if train_config.get("proj_beta_stair_bool", False) and proj_beta_stair_size:
                    stair_idx = min(proj_beta_stair_size_idx, len(proj_beta_stair_size) - 1)
                    step = float(proj_beta_stair_size[stair_idx])
                    logger.info(
                        f"proj_beta {model.layers[0].basic.get('proj_beta', 0.0)} has reached Pred threshold {pred_threshold}!"
                    )
                    for i, single_layer in enumerate(model.layers):
                        single_layer.basic["proj_beta"] = single_layer.basic.get("proj_beta", 0.0) + step
                        logger.info(
                            f"proj_beta of layer{i + 1} increased {step} to {single_layer.basic['proj_beta']}"
                        )
                    proj_beta_stair_size_idx += 1
            else:
                break

    train_total_elapsed = time.time() - train_total_start
    logger.info(f"训练总用时（epoch阶段）: {train_total_elapsed:.3f}s")


# 定义主函数，作为命令行入口
@app.command()
def main(
    # 配置文件路径，使用 --file 或 -f 参数指定
    config_file: Annotated[
        str, typer.Option("--file", "-f")
    ] = "./Configuration/2bit.toml",
    # 作业 ID，使用 --job-id 或 -j 参数指定
    job_id: Annotated[Optional[str], typer.Option("--job-id", "-j")] = None,
    # 覆盖配置，支持多次传入，例如 --override simulation.backend="fdfd_solver"
    override: Annotated[
        Optional[list[str]],
        typer.Option("--override", "-o"),
    ] = None,
):
    # 加载配置文件
    config = toml.load(config_file)
    applied_overrides = []
    if override:
        applied_overrides = apply_overrides(config, override)

    # 获取训练配置
    train_config = config["train"]
    enable_jit = bool(train_config.get("enable_jit", False))
    # 获取保存和加载配置
    save_load_config = config["save_load"]

    # 获取当前时间
    t = time.asctime(time.localtime(time.time()))
    # 获取当前工作目录
    pwd = os.getcwd()
    # 创建保存目录
    os.makedirs(f"{save_load_config['base_path']}/{t}", exist_ok=True)
    # 复制配置文件到保存目录
    config_dump_path = f"{save_load_config['base_path']}/{t}/config.toml"
    if applied_overrides:
        with open(config_dump_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    else:
        shutil.copyfile(config_file, config_dump_path)
    # 添加日志文件
    logger.add(f"{save_load_config['base_path']}/{t}/train.log")
    # 如果指定了作业 ID，记录到日志
    if job_id:
        logger.info(f"Slurm 作业 ID: {job_id}")
    # 记录配置文件路径
    logger.info(f"配置文件: {config_file}")
    logger.info(f"训练 JIT: {'开启' if enable_jit else '关闭'}")
    simulation_cfg = config.get("simulation", {}) or {}
    if (
        simulation_cfg.get("backend") == "fdfd_solver"
        and bool(simulation_cfg.get("enable_superlu_dist", False))
    ):
        dist_cfg = simulation_cfg.get("superlu_dist", {}) or {}
        native_complex = bool(dist_cfg.get("native_complex", True))
        logger.info(f"superlu_dist.native_complex: {native_complex}")
    if applied_overrides:
        logger.info(f"命令行覆盖参数: {applied_overrides}")
    # 记录开始训练时间和工作目录
    logger.info(f"开始训练于 {t} 在 {pwd}")
    # 记录保存目录
    logger.info(f"保存到: {save_load_config['base_path']}/{t}")

    task_type = _get_task_type(config)
    logger.info(f"训练任务类型: {task_type}")

    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    run_dir = f"{save_load_config['base_path']}/{t}"

    if task_type == "wdm_demux":
        _train_wdm(config, pwd, run_dir, ckptr)
    elif task_type == "sdm_demux":
        _train_sdm(config, pwd, run_dir, ckptr)
    else:
        _train_standard(config, pwd, run_dir, ckptr)

    ckptr.wait_until_finished()


# 如果是主程序，运行应用程序
if __name__ == "__main__":
    app()
