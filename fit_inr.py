import argparse
import logging
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from modula.atom import Linear
from modula.bond import ReLU
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.common.optimizers import get_lr, get_optimizer
from src.INR.modules import FourierFeats, LinearSimple
from src.INR.utils import get_grid

parser = argparse.ArgumentParser(description="Train videofitting")
parser.add_argument("-c", "--config", type=str, help="path to config .yaml")
args = parser.parse_args()
config_path = args.config

config = OmegaConf.load(config_path)

batch_size = config["batch_size"]
H = config["img_height"]
W = config["img_width"]
C = config["img_num_channels"]
width = config["mlp_width"]
steps = config["train_steps"]
seed = config["seed"]
val_freq = config["val_freq"]
folder = Path(config["out_folder"])
img_path = config["img_path"]
config_opt = config["optimizer"]

folder.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, folder / "config.yaml")

log = logging.getLogger(__name__)
logging.basicConfig(
    filename=folder / "log.txt",
    filemode="w",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

img = Image.open(img_path).resize((H, W))
if C == 1:
    img = img.convert("L")
elif C == 3:
    img = img.convert("RGB")
else:
    raise ValueError(f"wtf is this channel number {C}")
target = jnp.array(img) / 127.5 - 1
grid = get_grid(*target.shape[:2], b=1)
grid = grid * 2 - 1
inputs = grid.reshape(-1, 2)
targets = target[None].reshape(-1, C)
input_dim = inputs.shape[-1]
output_dim = targets.shape[-1]

mlp = Linear(output_dim, width)
mlp @= ReLU()
mlp @= Linear(width, width)
mlp @= ReLU()
mlp @= Linear(width, input_dim * 2)
mlp @= FourierFeats(input_dim, input_dim)

print(mlp)

mlp.jit()


def mse(w, inputs, targets):
    outputs = mlp(inputs, w)
    loss = ((outputs - targets) ** 2).mean()
    return loss


mse_and_grad = jax.jit(jax.value_and_grad(mse))

key = jax.random.PRNGKey(seed)
w = mlp.initialize(key)
optim = get_optimizer(config_opt)
opt_state = optim.init_state(w)


for step in tqdm(range(steps)):
    key, subkey = jax.random.split(key)
    idxs = jax.random.randint(subkey, (batch_size,), minval=0, maxval=inputs.shape[0])
    batch_inputs, batch_targets = inputs[idxs], targets[idxs]
    # compute loss and gradient of weights
    loss, grad_w = mse_and_grad(w, batch_inputs, batch_targets)

    if config.pre_dual:  # like in https://github.com/Arongil/lipschitz-transformers/blob/main/trainer.py#L50
        grad_w = mlp.dualize(grad_w)

    # compute scheduled learning rate
    lr = get_lr(schedule=config_opt.schedule, lr=config_opt.lr, step=step, steps=steps)

    _, opt_state, updates = optim.update(w, grad_w, opt_state)
    if config.post_dual:  # like in https://github.com/Arongil/lipschitz-transformers/blob/main/trainer.py#L57
        updates = mlp.dualize(updates)

    max_update_norm = lr
    w_decayed = [weight * (1 - config_opt.wd * max_update_norm) for weight in w]
    w = [weight_decayed - lr * update for weight_decayed, update in zip(w_decayed, updates)]
    # # update weights
    # w = [weight - lr * d_weight for weight, d_weight in zip(w, d_w)]

    if step % val_freq == 0:
        log.info(f"Step {step:3d} \t Loss {loss:.6f}")
        gen_img_np = np.array(mlp(inputs, w).reshape(H, W, C))
        np.clip(gen_img_np, -1, 1)
        gen_img_np = ((gen_img_np / 2 + 0.5) * 255).astype(np.uint8)
        Image.fromarray(gen_img_np).save(folder / f"{step}.jpg")

gen_img_np = np.array(mlp(inputs, w).reshape(H, W, C))
gen_img_np = np.clip(gen_img_np, -1, 1)
gen_img_np = ((gen_img_np / 2 + 0.5) * 255).astype(np.uint8)
Image.fromarray(gen_img_np).save(folder / f"{step}.jpg")

target_img_np = np.array(targets.reshape(H, W, C))
target_img_np = ((target_img_np / 2 + 0.5) * 255).astype(np.uint8)
Image.fromarray(target_img_np).save(folder / "target.jpg")
