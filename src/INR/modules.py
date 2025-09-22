import jax
import jax.numpy as jnp
from modula.abstract import Atom, Bond
from modula.atom import orthogonalize


class Sine(Bond):
    """Sine."""

    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return jnp.sin(self.sensitivity * x)


class SinCos(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        sin = jnp.sin(self.sensitivity * x)
        cos = jnp.cos(self.sensitivity * x)
        return jnp.concat([sin, cos], axis=-1)


class FourierFeats(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        return jnp.concat([jnp.sin((2.0 * jnp.pi * x) @ w[0].T), jnp.cos((2.0 * jnp.pi * x) @ w[0].T)], axis=-1)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = orthogonalize(grad) * jnp.sqrt(self.fanout / self.fanin) * target_norm
        return [d_weight]


class LinearSimple(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weights, x)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        return [grad]
