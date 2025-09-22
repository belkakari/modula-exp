# Modula library experiments
[Modula](https://docs.modula.systems/) is a library focused on exploring metrized deep learning and optimal feature learning in general. It is based on the nice paper [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)

## Setup
```bash
pip install uv
uv venv --python=3.12  # or whichever you like
source .venv/bin/activate
uv sync
```

## INR MLP
To run a simple INR MLP training
```bash
python fit_inr.py -c src/INR/config.yaml
```
### Results
| input                             | with gradient dualization           | without gradient dualization             | adam                           | muon                           |
| --------------------------------- | ----------------------------------- | ---------------------------------------- | ------------------------------ | ------------------------------ |
| ![input](./static/inr_target.jpg) | ![dual grad](./static/inr_dual.jpg) | ![no dual grad](./static/inr_simple.jpg) | ![adam](./static/inr_adam.jpg) | ![muon](./static/inr_muon.jpg) |

## DiT
Based on [DiT JAX implementation](https://github.com/kvfrans/jax-diffusion-transformer), [modula ViT PR](https://github.com/modula-systems/modula/pull/10) and [modula GPT example](https://docs.modula.systems/examples/hello-gpt/)

## Develop
Don't forget to set up pre-commit before commiting
```bash
pre-commit install
```
