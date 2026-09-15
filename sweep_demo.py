import sys

import os


import torch
# Force Python to search the local project directory first
project_root = "project_root"  # Update this to your project path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ultralytics.nn.tasks as tasks

# Change the sharing strategy to avoid saturating file descriptors
torch.multiprocessing.set_sharing_strategy('file_system')


# Force deterministic algorithms across PyTorch
torch.use_deterministic_algorithms(True, warn_only=False)

os.environ['PYTHONHASHSEED'] = '1'

project_root = os.environ.get("YOLO_DINOV2_PROJECT_ROOT", "/path/to/project")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Default sweep values
DEFAULTS = {
    "lr0": 1e-4,
    "box_weight": 7.5,
    "cls_weight": 0.5,
    "dfl_weight": 1.5,
}

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        wandb.init(
            project="YOLO-DINO_wandb_runs",
            job_type="training",
            config=DEFAULTS,
            mode="online",
        )
        config = dict(wandb.config)

        # Broadcast the config to the other ranks via environment variables.
        import json
        os.environ["SWEEP_CONFIG"] = json.dumps(config)
    else:
        # Secondary ranks read the config from the parent process.
        import json
        import time
        # Wait until rank 0 writes the config.
        for _ in range(30):
            if "SWEEP_CONFIG" in os.environ:
                break
            time.sleep(0.5)
        config = json.loads(os.environ.get("SWEEP_CONFIG", json.dumps(DEFAULTS)))

    if local_rank == 0:
        print(f"Sweep configuration: lr0={config['lr0']}, box={config['box_weight']}, "
              f"cls={config['cls_weight']}, dfl={config['dfl_weight']}")

    model = YOLO("model_root")

    model.train(
        data="dataset_root",
        epochs=50,
        imgsz=518,
        patience=15,
        batch=32,
        device=None,
        lr0=float(config["lr0"]),
        box=float(config["box_weight"]),
        cls=float(config["cls_weight"]),
        dfl=float(config["dfl_weight"]),
        fraction=1,
        freeze=1,
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

    if local_rank == 0 and wandb.run is not None:
        # Extract the metrics recorded in the Ultralytics summary.
        map50 = wandb.run.summary.get("metrics/mAP50(B)", 0.0)
        map50_95 = wandb.run.summary.get("metrics/mAP50-95(B)", 0.0)

        # Compute the combined metric.
        custom_score = 0.1 * map50 + 0.9 * map50_95

        # Save it directly in the summary so the sweep can read it without step issues.
        wandb.run.summary["metrics/custom_score"] = custom_score