import jax.numpy as jnp
from modula.abstract import Atom


class Bias(Atom):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [d]
        return weights

    def initialize(self, key):
        return [jnp.zeros(shape=self.d)]

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight) * jnp.sqrt(self.d)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        d_weight = grad / jnp.linalg.norm(grad) * jnp.sqrt(self.d) * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]


class Scale(Atom):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [d]
        return weights * x

    def initialize(self, key):
        return [jnp.ones(shape=self.d)]

    def project(self, w):
        weight = w[0]
        return [jnp.sign(weight)]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        return [jnp.sign(grad) * target_norm]
