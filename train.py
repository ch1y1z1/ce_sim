import os
import re
import shutil
from typing import Annotated, Any, Optional
import typer
import time
import toml
from model import CEmodel
import orbax.checkpoint as ocp
from flax import nnx
from loguru import logger
from dataset import prepare_io_dataset, decode_output
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax


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

    # 创建异步检查点器
    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    # 如果指定了加载路径，加载模型和优化器
    if "load_path" in save_load_config:
        logger.info(f"加载模型和优化器从: {save_load_config['load_path']}")
        # 创建模型
        model = CEmodel(config)
        # 分离模型的图定义和状态
        graphdef, state = nnx.split(model)
        # 从检查点加载状态
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/model", state)
        # 合并图定义和状态
        model = nnx.merge(graphdef, state)

        # 创建优化器
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))
        # 分离优化器的图定义和状态
        graphdef, state = nnx.split(optimizer)
        # 从检查点加载状态
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/optimizer", state)
        # 合并图定义和状态
        optimizer = nnx.merge(graphdef, state)
        # 修复：设置优化器的模型
        optimizer.model = model

    # 如果没有指定加载路径，初始化模型和优化器
    else:
        logger.info("初始化模型从零开始")
        # 创建模型
        model = CEmodel(config)
        logger.info(f"初始化优化器以步长: {train_config['step_size']}")
        # 创建优化器
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

    # 准备输入输出数据集
    n_bit, masks, expected_output, raw_i, raw_o = prepare_io_dataset(config["dataset"])

    # 定义损失函数
    # 参数：
    #   model: CEmodel 实例
    #   masks: 输入掩码矩阵
    # 返回：
    #   mse: 均方误差
    #   pred: 模型预测输出
    def loss_fn(model, masks):
        # 计算模型预测输出
        pred = model(masks)
        # 计算均方误差
        return jnp.mean(jnp.sum((pred - expected_output) ** 2, axis=1)), pred

    # 定义单步训练函数
    # 参数：
    #   model: CEmodel 实例
    #   optimizer: 优化器实例
    #   masks: 输入掩码矩阵
    # 返回：
    #   mse: 均方误差
    #   accu: 分类准确率
    def train_step_impl(model, optimizer, masks):
        # 计算损失函数的梯度
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        # 计算损失函数值和梯度
        (mse, pred), grads = grad_fn(model, masks)
        # 计算预测输出的分类标签
        pred_01 = jnp.where(pred > 0.5, 1.0, 0.0)
        pred_label = jax.vmap(decode_output)(pred_01)
        # 计算分类准确率
        accu = jnp.mean(pred_label == raw_o.astype(int))
        # 更新优化器
        optimizer.update(grads)
        # 返回均方误差和分类准确率
        return mse, accu

    train_step = nnx.jit(train_step_impl) if enable_jit else train_step_impl

    # 训练总计时：从首个 epoch 开始前计时
    train_total_start = time.time()

    # 训练模型
    for epoch in range(train_config["num_epochs"]):
        # 记录开始时间
        t_start = time.time()
        # 训练单步
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
        # 记录结束时间
        t_elapsed = time.time() - t_start
        # 记录训练信息
        logger.info(
            f"第 {epoch} 个epoch，均方误差: {mse}, 分类准确率: {accu}, 时间: {t_elapsed}s"
        )

        # 如果需要保存模型，保存模型和优化器
        if (
            save_load_config["save"]
            and (epoch - 1) % save_load_config["save_interval"] == 0
        ):
            # 分离模型的图定义和状态
            _, state = nnx.split(model)
            # 保存模型状态
            ckptr.save(
                f"{pwd}/{save_load_config['base_path']}/{t}/models/epoch_{epoch}/model",
                # args=ocp.args.StandardSave(state),
                args=ocp.args.PyTreeSave(state),
            )
            # 分离优化器的图定义和状态
            _, state = nnx.split(optimizer)
            # 保存优化器状态
            ckptr.save(
                f"{pwd}/{save_load_config['base_path']}/{t}/models/epoch_{epoch}/optimizer",
                args=ocp.args.PyTreeSave(state),
            )

        # 如果需要可视化，保存可视化结果
        if (
            "viz_interval" in save_load_config
            and (epoch - 1) % save_load_config["viz_interval"] == 0
        ):
            # 创建可视化目录
            os.makedirs(
                f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch}", exist_ok=True
            )
            # 遍历输入掩码矩阵
            for ndx, mask in enumerate(masks):
                # 计算模型预测输出
                fig, _ = model.viz_abs(mask)
                # 保存可视化结果
                plt.savefig(
                    f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch}/{raw_i[ndx]}_{raw_o[ndx]}.png"
                )
                # 关闭图像
                plt.close(fig)

    train_total_elapsed = time.time() - train_total_start
    logger.info(f"训练总用时（epoch阶段）: {train_total_elapsed:.3f}s")

    # 等待检查点器完成
    ckptr.wait_until_finished()


# 如果是主程序，运行应用程序
if __name__ == "__main__":
    app()
