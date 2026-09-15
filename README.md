# YOLO-DINOv2 Deterministic

This repository combines a DINOv2 Vision Transformer backbone with Ultralytics YOLO detection heads for reproducible object detection experiments. It includes the custom DINOv2 patch module, deterministic training scripts, visualization and validation utilities, and YAML model definitions used in this project.

The project is oriented toward research and experiment reproducibility. Local dataset paths, checkpoints, and external research assets are not bundled in this repository and must be configured in the local environment.

## Highlights

- DINOv2 backbones in multiple sizes: `small`, `base`, `large`, and `giant`.
- Custom `DinoV2Patches` integration that converts DINOv2 patch tokens into a YOLO-compatible feature map.
- Deterministic PyTorch execution in the training scripts.
- Project-specific scripts for training, W&B sweeps, validation, and inference.
- Ultralytics-based model definitions and experiment tracking workflows.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ultralytics/nn/modules/pretrained_vit.py` | DINOv2 backbone and patch-token conversion logic |
| `ultralytics/nn/tasks.py` | Ultralytics task registration for the custom module |
| `yolo_dinov2_configs/` | YOLO model configuration files built around DINOv2 backbones |
| `demo.py` | Main deterministic training script with W&B logging |
| `sweep_demo.py` | Sweep-based hyperparameter training example |
| `inference.py` | Inference and image-level evaluation workflow |
| `validation.py` | Bootstrap-style validation experiment script |
| `BootstrapValidator.py` | Custom validation/bootstrap utility |
| `runs/` | Saved training outputs and artifacts |
| `wandb/` | Local Weights & Biases metadata from experiments |
| `tests/` | Ultralytics regression and validation tests |

## Requirements

- Python 3.8+
- PyTorch and torchvision
- CUDA-capable GPU recommended for training and heavy validation runs
- Local checkout of the DINOv2 source tree
- A compatible RAD-DINO or DINOv2 checkpoint in SafeTensors format
- An Ultralytics-style dataset YAML file for training or validation

Install the project dependencies from the repository requirements file:

```bash
git clone https://github.com/<YOUR-USER>/<YOUR-REPOSITORY>.git
cd project-name
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development work, you can also install the project in editable mode and additional dev dependencies:

```bash
python -m pip install -e .
python -m pip install .[dev]
```

## External model assets

`DinoV2Patches` expects a local DINOv2 checkout and a compatible backbone checkpoint. Before running training or inference, edit the paths in the scripts for your environment. Typical entries look like:

```python
DINO_PATH = "/path/to/dinov2"
CHECKPOINT_PATH = "/path/to/backbone_compatible.safetensors"
```

The repository currently contains example research paths and should be treated as local configuration rather than portable defaults. Do not commit private datasets, personal checkpoints, or credentials to version control.

## Model configurations

The model definitions live in `yolo_dinov2_configs/`:

- `yolo_dinov2_small.yaml`
- `yolo_dinov2_large.yaml`
- `yolo_dinov2_giant.yaml`
- `yolon_dinov2.yaml`
- `apple_aimv2_large.yaml`

Set `nc` in the selected YAML file to the number of classes in your dataset. The small configuration produces a 384-channel feature map, while the larger variants use 768, 1024, or 1536 channels.

## Deterministic training

The training scripts set deterministic behavior in PyTorch and fix `PYTHONHASHSEED` before running. A minimal pattern is:

```python
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches

os.environ["PYTHONHASHSEED"] = "1"
torch.use_deterministic_algorithms(True, warn_only=False)
tasks.DinoV2Patches = DinoV2Patches

model = YOLO("yolo_dinov2_configs/yolo_dinov2_small.yaml")
model.train(
    data="/path/to/dataset.yaml",
    epochs=50,
    imgsz=518,
    batch=32,
    seed=1,
    deterministic=True,
    pretrained=False,
)
```

Run the project scripts with:

```bash
python demo.py
python sweep_demo.py
```

Both scripts are configured for local research runs and may need path updates for your dataset, DINOv2 checkout, and GPU devices.

## Validation and inference

A typical validation flow is:

```python
from ultralytics import YOLO

model = YOLO("/path/to/best.pt")
model.val(
    data="/path/to/dataset.yaml",
    split="test",
    batch=32,
    device=0,
)
```

For the repository-specific local workflows, edit the paths in `inference.py` or `validation.py` to match your dataset and model outputs, then run:

```bash
python inference.py
python validation.py
```

These scripts are tailored to the project’s own validation setup and include custom visualization or bootstrap-based evaluation logic.

## Experiment tracking

The project is configured to log runs to Weights & Biases. Authenticate once and then run your selected training script:

```bash
wandb login
python demo.py
```

You can also use the sweep script for hyperparameter exploration:

```bash
python sweep_demo.py
```

Disable or remove the W&B integration when working offline or in a restricted environment.

## Testing

Run the lightweight project tests with:

```bash
pytest -q
```

Some tests depend on extra packages, downloaded assets, or a CUDA-enabled device. Run the relevant module when debugging a specific subsystem.

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- RAD-DINO and related research work for compatible backbone weights and experimental setup

## License

This repository includes and modifies code from Ultralytics. Review the [Ultralytics AGPL-3.0 license](LICENSE) and the licenses of the external DINOv2 and RAD-DINO components before redistribution or commercial use. Cite the upstream projects and the research work that produced any included weights.

## Contributing

Issues and pull requests are welcome. Please include the command, environment, dataset assumptions, and a concise reproduction when reporting a bug or experiment issue.
