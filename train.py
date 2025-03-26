import os
import shutil
from typing import Annotated, Optional
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


app = typer.Typer()


@app.command()
def main(
    config_file: Annotated[
        str, typer.Option("--file", "-f")
    ] = "./Configuration/2bit.toml",
    job_id: Annotated[Optional[str], typer.Option("--job-id", "-j")] = None,
):
    config = toml.load(config_file)
    train_config = config["train"]
    save_load_config = config["save_load"]

    t = time.asctime(time.localtime(time.time()))
    pwd = os.getcwd()
    os.makedirs(f"{save_load_config['base_path']}/{t}", exist_ok=True)
    shutil.copyfile(config_file, f"{save_load_config['base_path']}/{t}/config.toml")
    logger.add(f"{save_load_config['base_path']}/{t}/train.log")
    if job_id:
        logger.info(f"Slurm job ID: {job_id}")
    logger.info(f"Configuration file: {config_file}")
    logger.info(f"Starting training at {t} in {pwd}")
    logger.info(f"Saving to: {save_load_config['base_path']}/{t}")

    ckptr = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    if "load_path" in save_load_config:
        logger.info(
            f"Loading model and optimizer from: {save_load_config['load_path']}"
        )
        model = CEmodel(config)
        graphdef, state = nnx.split(model)
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/model", state)
        model = nnx.merge(graphdef, state)

        # important: load optimizer state!
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))
        graphdef, state = nnx.split(optimizer)
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}/optimizer", state)
        optimizer = nnx.merge(graphdef, state)
        # FIX:
        optimizer.model = model

    else:
        logger.info("Initializing model from scratch")
        model = CEmodel(config)
        logger.info(
            f"Initializing optimizer with step size: {train_config['step_size']}"
        )
        optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"]))

    n_bit, masks, expected_output, raw_i, raw_o = prepare_io_dataset(config["dataset"])

    def loss_fn(model, masks):
        pred = model(masks)
        return jnp.mean(jnp.sum((pred - expected_output) ** 2, axis=1)), pred

    def train_step(model, optimizer, masks):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (mse, pred), grads = grad_fn(model, masks)
        pred_01 = jnp.where(pred > 0.5, 1.0, 0.0)
        pred_label = jax.vmap(decode_output)(pred_01)
        accu = jnp.mean(pred_label == raw_o.astype(int))
        optimizer.update(grads)
        return mse, accu

    for epoch in range(train_config["num_epochs"]):
        t_start = time.time()
        mse, accu = train_step(model, optimizer, masks)
        t_elapsed = time.time() - t_start
        logger.info(
            f"Epoch {epoch}/{train_config['num_epochs']}, MSE: {mse}, Accu: {accu}, Time: {t_elapsed}s"
        )

        if (
            save_load_config["save"]
            and (epoch - 1) % save_load_config["save_interval"] == 0
        ):
            _, state = nnx.split(model)
            ckptr.save(
                f"{pwd}/{save_load_config['base_path']}/{t}/models/epoch_{epoch}/model",
                # args=ocp.args.StandardSave(state),
                args=ocp.args.PyTreeSave(state),
            )
            # important: save the optimizer state as well
            _, state = nnx.split(optimizer)
            ckptr.save(
                f"{pwd}/{save_load_config['base_path']}/{t}/models/epoch_{epoch}/optimizer",
                args=ocp.args.PyTreeSave(state),
            )

        if (
            "viz_interval" in save_load_config
            and (epoch - 1) % save_load_config["viz_interval"] == 0
        ):
            os.makedirs(
                f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch}", exist_ok=True
            )
            for ndx, mask in enumerate(masks):
                fig, _ = model.viz_abs(mask)
                # save
                plt.savefig(
                    f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch}/{raw_i[ndx]}_{raw_o[ndx]}.png"
                )
                plt.close(fig)

    ckptr.wait_until_finished()


if __name__ == "__main__":
    app()
