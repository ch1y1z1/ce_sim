import os
import shutil
from typing import Annotated
import typer
import time
import toml
from model import CEmodel
import orbax.checkpoint as ocp
from flax import nnx
from loguru import logger
from dataset import prepare_io_dataset
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt


app = typer.Typer()

@app.command()
def main(
        config_file: Annotated[str, typer.Option("--file", "-f")] = "./Configuration/2bit.toml",
        job_id: Annotated[str, typer.Option("--job-id", "-j")] = None
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


    model = CEmodel(config)
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    if "load_path" in save_load_config:
        logger.info(f"Loading model from: {save_load_config['load_path']}")
        graphdef, state = nnx.split(model)
        state = ckptr.restore(f"{pwd}/{save_load_config['load_path']}", state)
        model = nnx.merge(graphdef, state)

    n_bit, masks, expected_output, raw_i, raw_o = prepare_io_dataset(config["dataset"])
    optimizer = nnx.Optimizer(model, optax.adamw(train_config["step_size"], 0.9))

    def loss_fn(model, masks):
        pred = model(masks)
        return jnp.sum((pred - expected_output) ** 2)

    def train_step(model, optimizer, masks):
        grad_fn = nnx.value_and_grad(loss_fn)
        mse, grads = grad_fn(model, masks)
        optimizer.update(grads)
        return mse

    for epoch in range(train_config["num_epochs"]):
        t_start = time.time()
        mse = train_step(model, optimizer, masks)
        t_elapsed = time.time() - t_start
        logger.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}, MSE: {mse}, Time: {t_elapsed}s")

        if save_load_config["save"] == True and epoch % save_load_config["save_interval"] == 0:
            _, state = nnx.split(model)
            ckptr.save(f"{pwd}/{save_load_config['base_path']}/{t}/models/epoch_{epoch + 1}", args=ocp.args.StandardSave(state))

        if "viz_interval" in save_load_config and epoch % save_load_config["viz_interval"] == 0:
            os.makedirs(f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch + 1}", exist_ok=True)
            for ndx, mask in enumerate(masks):
                fig, _ = model.viz_abs(mask)
                # save
                plt.savefig(f"{save_load_config['base_path']}/{t}/viz/epoch_{epoch + 1}/{raw_i[ndx]}_{raw_o[ndx]}.png")
                plt.close(fig)

if __name__ == "__main__":
    app()
    # main(
    #     config_file="./Configuration/2bit.toml"
    # )
