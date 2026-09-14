# YOLO-DINOv2 Deterministic

This repository combines an Ultralytics YOLO detection head with a DINOv2 Vision Transformer backbone for object detection. It contains the custom `DinoV2Patches` module, model configuration files, deterministic training helpers, validation scripts, and analysis utilities used in the project experiments.

The code is intended for research and reproducible experimentation. Dataset files, trained checkpoints, and RAD-DINO weights are not included in this repository.

## Highlights

- DINOv2 register backbones in four sizes: `small`, `base`, `large`, and `giant`.
- A custom `DinoV2Patches` layer that converts DINOv2 patch tokens into a feature map for the YOLO detection head.
- Deterministic PyTorch settings in the training examples and the local Ultralytics utilities.
- Standard Ultralytics training, validation, prediction, and experiment logging workflows.
- Optional Weights & Biases integration through `demo.py` and `sweep_demo.py`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ultralytics/nn/modules/pretrained_vit.py` | DINOv2 backbone integration and patch-token conversion |
| `yolo_dinov2_configs/` | YOLO model definitions using `DinoV2Patches` |
| `demo.py` | Deterministic training example |
| `sweep_demo.py` | Deterministic W&B sweep example |
| `validation_onerun.py` | Validation of a trained checkpoint |
| `inference_copy.py` | Single-image inference with custom colored labels |
| `BootstrapValidator.py` and `paired_bootstraping_comparison.py` | Bootstrap-based evaluation utilities |
| `class_weights.py` and `weighted_dataset.py` | Class balancing helpers |
| `tests/` | Ultralytics regression and integration tests |

## Requirements

- Python 3.8 or newer
- PyTorch and torchvision
- A CUDA-capable GPU is recommended for DINOv2 training and inference
- A local checkout of the DINOv2 repository
- A compatible RAD-DINO checkpoint in SafeTensors format
- An Ultralytics-format detection dataset YAML file

The project uses the dependencies declared in [`pyproject.toml`](pyproject.toml). Install the local package in editable mode:

```bash
git clone https://github.com/<YOUR-USER>/<YOUR-REPOSITORY>.git
cd Yolo-DinoV2-deterministic-copy
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## External model assets

`DinoV2Patches` loads the DINOv2 architecture through `torch.hub` and then loads a RAD-DINO-compatible checkpoint from SafeTensors. Before running a script, update the paths in the script and in [`ultralytics/nn/modules/pretrained_vit.py`](ultralytics/nn/modules/pretrained_vit.py) for your environment:

```python
DINO_PATH = "/path/to/dinov2"
CHECKPOINT_PATH = "/path/to/backbone_compatible.safetensors"
```

The current source contains paths from the original research environment. They are examples, not portable defaults. Do not commit private datasets, checkpoints, or credentials to GitHub.

## Model configurations

Available configurations are in [`yolo_dinov2_configs/`](yolo_dinov2_configs/):

- `yolo_dinov2_small.yaml`
- `yolo_dinov2_large.yaml`
- `yolo_dinov2_giant.yaml`
- `yolon_dinov2.yaml`
- `apple_aimv2_large.yaml`

Set `nc` in the selected YAML file to the number of classes in your dataset. The default small configuration uses a 384-channel DINOv2 feature map; the other DINOv2 variants use 768, 1024, or 1536 output channels respectively.

## Deterministic training

The training examples enable deterministic PyTorch algorithms and set `PYTHONHASHSEED`. A minimal training pattern is:

```python
import os
import sys
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches
import ultralytics.nn.tasks as tasks

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

Or adapt the project defaults in `demo.py`:

```bash
python demo.py
```

Exact reproducibility can still vary with GPU hardware, CUDA/cuDNN versions, multiprocessing, and the data-loading configuration. Record the commit, environment, dataset revision, checkpoint, seed, device, and training arguments for each experiment.

## Validation and inference

Validate a trained model with the Ultralytics API:

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

For the repository's custom single-image rendering, configure `MODEL_PATH`, `IMAGE_PATH`, `DINO_PATH`, and `SAVE_DIR` in `inference_copy.py`, then run:

```bash
python inference_copy.py
```

The script resizes the image to 640x640, applies the configured confidence and IoU thresholds, draws class-specific boxes, and saves a PDF prediction next to the configured output directory.

## Experiment tracking

`demo.py` and `sweep_demo.py` can log training runs to Weights & Biases. Configure the W&B project and authenticate before running:

```bash
wandb login
python demo.py
```

Disable or remove the W&B integration when working offline.

## Testing

Run the lightweight test suite with:

```bash
pytest -q
```

Some tests require additional packages, downloaded assets, or a CUDA device. Run the relevant test module when working on a specific subsystem.

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research.
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework.
- RAD-DINO and the associated medical-imaging research that supplied compatible backbone weights.

## License

This repository includes and modifies code from Ultralytics. Review the [Ultralytics AGPL-3.0 license](LICENSE) and the licenses of the external DINOv2 and RAD-DINO components before redistribution or commercial use. Cite the upstream projects and the research work that produced any included weights.

## Contributing

Issues and pull requests are welcome. Please include the command, environment, dataset/configuration assumptions, and a concise reproduction when reporting a problem.