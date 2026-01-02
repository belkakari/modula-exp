import jax.numpy as jnp
from modula.atom import Embed, Linear
from modula.bond import ApplyAttentionScores, AttentionQK, CausalMask, GeLU, MergeHeads, Rope, Softmax, SplitIntoHeads

from src.common.atoms import Bias
from src.common.bonds import Patchify, SiLU, TimestepEmb


def AttentionViT(num_heads, d_embed, d_query, d_value, softmax_scale, causal, posemb="rope", bias=False):
    """
    Multi-head attentio
    Taken from https://github.com/modula-systems/modula/blob/b33055e95db27b8a0e03079bf28195ade46bd5a4/modula/compound.py#L13-L28
    """
    Q, K, V = (
        Linear(num_heads * d_query, d_embed),
        Linear(num_heads * d_query, d_embed),
        Linear(num_heads * d_value, d_embed),
    )
    Q = SplitIntoHeads(num_heads) @ (Q + Bias(num_heads * d_query) if bias else Q)
    K = SplitIntoHeads(num_heads) @ (K + Bias(num_heads * d_query) if bias else K)
    V = SplitIntoHeads(num_heads) @ (V + Bias(num_heads * d_value) if bias else V)
    W = Linear(d_embed, num_heads * d_value)
    W = (W + Bias(d_embed) if bias else W) @ MergeHeads()
    QK = (Q, K)
    if posemb == "rope":
        QK = Rope(d_query) @ QK
    attn = AttentionQK() @ QK
    if causal:
        attn = CausalMask() @ attn
    AttentionScores = Softmax(softmax_scale) @ attn
    return W @ (1 / 3 * ApplyAttentionScores()) @ (V, AttentionScores)


def MLP(input_dim, output_dim, mlp_dim):
    mlp = Linear(output_dim, mlp_dim)
    mlp @= GeLU()
    mlp @= Linear(mlp_dim, input_dim)

    mlp.jit()

    return mlp


def PatchEmbed(d_embed, patch_size, num_channels, bias=False):
    p1, p2 = patch_size
    patchify = Linear(d_embed, p1 * p2 * num_channels) @ Patchify(patch_size)

    if bias:
        patchify = patchify + Bias(d_embed)

    patchify.jit()

    return patchify


def LabelEmbed(num_classes, d_embed):
    label_embed = Embed(d_embed=d_embed, num_embed=num_classes)

    return label_embed


def TimestepEmbed(frequency_embedding_size, hidden_size):
    ts_embedder = (
        Linear(hidden_size, hidden_size)
        @ SiLU()
        @ Linear(hidden_size, frequency_embedding_size)
        @ TimestepEmb(frequency_embedding_size=frequency_embedding_size)
    )
    return ts_embedder


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]
