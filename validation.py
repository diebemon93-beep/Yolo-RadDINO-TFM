import sys

import os


import torch

# Force deterministic algorithms throughout PyTorch
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ['PYTHONHASHSEED'] = '0'


# Force Python to search the local project directory first
project_root = "project_root"  # Update this to your project path
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Import the official installation tasks
import ultralytics.nn.tasks as tasks

from ultralytics import YOLO
from BootstrapValidator import FastBootstrapValidator
import numpy as np


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

# Force safety patches to avoid YAML format problems.
os.environ["WANDB_PLOT_BBOX_ARR"] = "false"
os.environ["WANDB_LOG_MODEL"] = "true"

# Force Python to find the 'dinov2' and 'rad_dino' modules.
DINO_PATH = "DINO_PATH"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

model = YOLO("path_to_best.pt")

# Execute N bootstrap iterations
N_ITER = 1000
N_SAMPLE = 291
USE_CONTAINMENT = True

results = []

print(
    f"USE_CONTAINMENT={USE_CONTAINMENT}; "
    f"matching criterion: {'predicted box contained within GT' if USE_CONTAINMENT else 'standard IoU'}"
)
METRIC_PREFIX = "Pseudo-" if USE_CONTAINMENT else ""
METRIC_MODE = "containment" if USE_CONTAINMENT else "IoU"
if USE_CONTAINMENT:
    print(
        "The mAP50 and mAP50-95 metrics use the same binary matching by containment; "
        "they do not represent different IoU thresholds."
    )

for i in range(N_ITER):

    # Force a reset of the cached validator
    if hasattr(model, 'validator'):
        del model.validator

    metrics = model.val(
        data='dataset_root',
        split='test',
        device=[2],
        batch=32,
        plots=True,
        save_json=True,
        verbose=True,
        use_containment=USE_CONTAINMENT,
        validator=lambda args, _callbacks=None: FastBootstrapValidator(
            args=args,
            _callbacks=_callbacks,
            n_bootstrap=N_SAMPLE,
        ),
    )
    results.append({
        "iter": i,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
    })
    print(f"Iteration {i+1}/{N_ITER}: {METRIC_PREFIX}mAP({METRIC_MODE})={metrics.box.map50:.4f}")
    print(f"Iteration {i+1}/{N_ITER}: {METRIC_PREFIX}mAP50-95({METRIC_MODE})={metrics.box.map:.4f}")
    print(f"Iteration {i+1}/{N_ITER}: {METRIC_PREFIX}Precision({METRIC_MODE})={metrics.box.mp:.4f}")
    print(f"Iteration {i+1}/{N_ITER}: {METRIC_PREFIX}Recall({METRIC_MODE})={metrics.box.mr:.4f}")

# Compute 95% confidence intervals
map50s = np.array([r["mAP50"] for r in results])
map_50_95s = np.array([r["mAP50-95"] for r in results])
precisions = np.array([r["Precision"] for r in results])
recalls = np.array([r["Recall"] for r in results])

print("\n" + "=" * 55)
print("FINAL BOOTSTRAP RESULTS (95% CI)")
print("=" * 55)

print(f"{METRIC_PREFIX}mAP({METRIC_MODE}) mean : {map50s.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}mAP({METRIC_MODE})    : [{np.percentile(map50s, 2.5):.4f}, {np.percentile(map50s, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}mAP50-95({METRIC_MODE}) mean : {map_50_95s.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}mAP50-95({METRIC_MODE}) : [{np.percentile(map_50_95s, 2.5):.4f}, {np.percentile(map_50_95s, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}Precision({METRIC_MODE}) mean : {precisions.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}Precision({METRIC_MODE}): [{np.percentile(precisions, 2.5):.4f}, {np.percentile(precisions, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}Recall({METRIC_MODE}) mean    : {recalls.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}Recall({METRIC_MODE})   : [{np.percentile(recalls, 2.5):.4f}, {np.percentile(recalls, 97.5):.4f}]")
print("=" * 55)