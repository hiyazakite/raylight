# Raylight

Raylight. Using Ray Worker to manage multi GPU sampler setup. With XDiT XFuser to implement sequence parallelism 

## Wan T2V 1.3B
<img width="1918" height="887" alt="image" src="https://github.com/user-attachments/assets/57b7cdf5-ebd5-4902-bccd-fa7bbfe9ef8b" />

https://github.com/user-attachments/assets/40deddd2-1a87-44de-98a5-d5fc3defbecd

## Wan T2V 14B on RTX 2000 ADA ≈ RTX 4060 TI 16GB
<img width="1117" height="716" alt="Screenshot 2025-08-20 125936" src="https://github.com/user-attachments/assets/b2ea1621-4be1-4925-8f4a-3ff9542c6415" />

## DEBUG Notes

### Wan T2V 1.3B (bf16) — 1×3090 24G

| Setup                        | VRAM (GB) |
|------------------------------|-----------|
| Non-Ulysses                  | 4.6       |
| Ulysses                      | 5.1       |
| XDiT Ring + Ulysses          | 9.2       |
| Ulysses + Sync Module FSDP1  | 9.7       |
| Ulysses + Unsync Module FSDP1| 5.2       |

---

### Wan T2V 14B (fp8) — 1×3090 24G

| Setup                        | VRAM (GB)        |
|------------------------------|------------------|
| Non-Ulysses                  | 15.5             |
| Ulysses                      | 15.9             |
| Ulysses + Sync Module FSDP1  | OOM (Pred. 47.3) |
| Ulysses + Unsync Module FSDP1| OOM (Pred. 31.8) |
| XDiT Ring + Ulysses          | OOM              |

---

### Wan T2V 14B (fp8) — 1×RTX 2000 ADA 16G
**Resolution:** 480×832 × 33F

| Setup   | VRAM (GB) | Speed            |
|---------|-----------|------------------|
| Normal  | OOM       | 22 it/s (before OOM) |

---

### Wan T2V 14B (fp8) — 2×RTX 2000 ADA 16G

| Setup           | VRAM (GB) / Device | Speed   |
|-----------------|--------------------|---------|
| Ulysses         | 15.8 (Near OOM)    | 11 it/s |
| FSDP2           | 12.8               | 19 it/s |
| Ulysses + FSDP2 | 10.25              | 12 it/s |

---

### Notes
- **FSDP OOM in single-device 14B model** caused by:
  - Sync Module FSDP: model is first loaded into CUDA, then sharded (not tested on dual GPU).
  - Lowest possible dtype for FSDP params is **bf16**, hence doubled size compared to fp8.
- **FSDP2** is now available and can do fp8 calculation, but needs scalar tensors converted into 1D tensors.


## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- A list of features

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd raylight
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Writing custom nodes

An example custom node is located in [node.py](src/raylight/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile).
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

