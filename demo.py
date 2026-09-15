import sys

import os


import torch

# Force deterministic algorithms across PyTorch
torch.use_deterministic_algorithms(True, warn_only=False)

os.environ['PYTHONHASHSEED'] = '1'


# Force Python to search the local project directory first
project_root = "project_root"  # Update this to your project path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Import the official installation tasks
import ultralytics.nn.tasks as tasks


# Import the custom block from your development folder
# Make sure the path to the 'DinoV2Patches' class is correct.
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches

# Inject the custom module into the official globals so parse_model can resolve it.
tasks.DinoV2Patches = DinoV2Patches
tasks.__dict__["DinoV2Patches"] = DinoV2Patches

import os
import wandb
from ultralytics import YOLO
# Use the native import so custom architectures are not broken.
from ultralytics.utils.callbacks import add_integration_callbacks

import ultralytics.data.build as build
from weighted_dataset import YOLOWeightedDataset

# Intercept the default Ultralytics class with the custom balanced class.
# build.YOLODataset = YOLOWeightedDataset
# print("YOLODataset successfully intercepted. Using YOLOWeightedDataset for balancing.")

# Force safety patches to avoid YAML format issues.
os.environ["WANDB_PLOT_BBOX_ARR"] = "false"
os.environ["WANDB_LOG_MODEL"] = "true"

# os.environ["WANDB_RUN_ID"] = "u9gmq55e"  # Replace with your original run ID
# os.environ["WANDB_RESUME"] = "must"  # Force W&B to reconnect to the existing run

DINO_PATH = "DINO_PATH"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    lr0 = 0.01
    batch_size = 32
    imgsz = 518
    box = 13.5  # (float) box loss gain
    cls_ = 0.8  # (float) cls loss gain (scale with pixels)
    dfl = 2.5

    if local_rank == 0:
        wandb.init(
            project="YOLO-DINO_wandb_runs",
            job_type="training",
            name="deterministic_sweep",
            resume="allow",
            config={
                "lr0": lr0,
                "batch": batch_size,
                "img_size": imgsz,
                "box_weight": box,
                "cls_weight": cls_,
                "dfl_weight": dfl,
            }
        )

    model = YOLO("model_root")

    # Moderate configuration version
    model.train(
        data="data_root",
        epochs=5,
        batch=batch_size,
        device=[4],
        lr0=lr0,
        imgsz=imgsz,
        project="YOLO-DINO_wandb_runs",
        name="lp_base_50_medium_sgd_adapted",
        fraction=1,
        dfl=dfl,
        cls=cls_,
        box=box,
        optimizer="SGD",
        momentum=0.9,
        cos_lr=True,
        warmup_bias_lr=0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        pretrained=False,
    )

    if local_rank == 0:
        wandb.finish()