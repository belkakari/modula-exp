import jax
import jax.numpy as jnp
from modula.abstract import Identity
from modula.atom import Linear
from modula.bond import GeLU

from src.common.atoms import Bias, Scale
from src.common.bonds import Constant, LayerNorm, Mean

from .modules import MLP, AttentionViT, LabelEmbed, PatchEmbed, TimestepEmbed, posemb_sincos_2d


def ViT(
    num_classes,
    image_size=(28, 28),
    patch_size=(7, 7),
    num_heads=4,
    d_embed=32,
    d_query=8,
    d_value=8,
    num_blocks=4,
    blocks_mass=5,
    attention_scale=1.0,
    final_scale=1.0,
    channels=1,
    LN=True,
    bias=True,
    scale=True,
    freq_emb=128,
):
    i1, i2 = image_size
    p1, p2 = patch_size
    h, w = i1 // p1, i2 // p2
    patchify = PatchEmbed(d_embed=d_embed, patch_size=patch_size)
    posemb = Constant(lambda: posemb_sincos_2d(h, w, d_embed))
    tsemb = TimestepEmbed(frequency_embedding_size=freq_emb, hidden_size=d_embed)
    att = AttentionViT(num_heads, d_embed, d_query, d_value, attention_scale, causal=False, posemb="none", bias=bias)
    mlp = (
        (Linear(d_embed, 4 * d_embed) + Bias(d_embed) if bias else Linear(d_embed, 4 * d_embed))
        @ GeLU()
        @ (Linear(4 * d_embed, d_embed) + Bias(4 * d_embed) if bias else Linear(4 * d_embed, d_embed))
    )
    if LN:
        ln = LayerNorm()
        if bias and scale:
            ln = (Scale(d_embed) + Bias(d_embed)) @ ln
        elif bias:
            ln = ln + Bias(d_embed)
        elif scale:
            ln = Scale(d_embed) @ ln
        att = att @ ln
        mlp = mlp @ ln
    att_block = (1 - 1 / (2 * num_blocks)) * Identity() + 1 / (2 * num_blocks) * att
    mlp_block = (1 - 1 / (2 * num_blocks)) * Identity() + 1 / (2 * num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    gap = Mean(axis=1, size=h * w)
    out = final_scale * (Linear(num_classes, d_embed) + Bias(num_classes) if bias else Linear(num_classes, d_embed))

    ret = blocks @ (patchify + posemb + tsemb)
    if LN:  # Final LN
        ret = ln @ ret
    return out @ gap @ ret
