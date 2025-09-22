import einops
import jax
import jax.numpy as jnp
from modula.abstract import Bond


class Constant(Bond):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.smooth = True
        self.sensitivity = 0

    def forward(self, x, w):
        return self.f()


class LayerNorm(Bond):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps)


class SiLU(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = False
        self.sensitivity = 1

    def forward(self, x, w):
        return jax.nn.silu(x) / 1.1289  # 1.1289 is the max derivative of gelu(x), keeping it here too


class Mean(Bond):
    def __init__(self, axis, size):
        super().__init__()
        self.smooth = True
        self.axis = axis
        self.size = size
        self.sensitivity = 1 / size

    def forward(self, x, w):
        assert x.shape[self.axis] == self.size
        return jax.numpy.mean(x, axis=self.axis)


class Patchify(Bond):
    def __init__(self, size):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
        self.size = size

    def forward(self, x, w):
        p1, p2 = self.size
        return einops.rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=p1, p2=p2)


class TimestepEmb(Bond):
    def __init__(self, frequency_embedding_size, max_period=10_000):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
        half = frequency_embedding_size // 2
        self.freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)

    def forward(self, x, w):
        args = x[:, None] * self.freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
