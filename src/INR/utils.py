import jax.numpy as jnp


def get_grid(h, w, b=1, norm=True, device="cpu"):
    if norm:
        xgrid = jnp.linspace(0, w, num=w) / w
        ygrid = jnp.linspace(0, h, num=h) / h
    else:
        xgrid = jnp.linspace(0, w, num=w)
        ygrid = jnp.linspace(0, h, num=h)
    xv, yv = jnp.meshgrid(xgrid, ygrid, indexing="xy")
    grid = jnp.stack([xv, yv], axis=-1)[None]

    return jnp.tile(grid, (b, 1, 1, 1))
