import sys

import os


import torch

# Forzar algoritmos deterministas en todo PyTorch
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ['PYTHONHASHSEED'] = '0'


# Forzar a Python a buscar primero en tu carpeta local del proyecto
project_root = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy"  # Cambia esto a la ruta de tu proyecto
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 2. IMPORTANTE: Importamos las tareas de la instalación oficial
import ultralytics.nn.tasks as tasks

from ultralytics import YOLO
from BootstrapValidator import FastBootstrapValidator
import numpy as np


# 3. Importamos TU bloque customizado desde tu carpeta de desarrollo
# (Asegúrate de que la ruta a tu clase 'DinoV2Patches' sea la correcta)
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches  

# 4. TRUCO MAGISTRAL: Inyectamos tu módulo en los globales oficiales
# Esto hace que 'parse_model' encuentre tu capa aunque use el paquete limpio.
tasks.DinoV2Patches = DinoV2Patches
tasks.__dict__["DinoV2Patches"] = DinoV2Patches

import os
import wandb
from ultralytics import YOLO
# Usamos el import nativo que no rompe arquitecturas personalizadas
from ultralytics.utils.callbacks import add_integration_callbacks

import ultralytics.data.build as build
from weighted_dataset import YOLOWeightedDataset

# Interceptamos la clase por defecto de Ultralytics con tu clase balanceada
# build.YOLODataset = YOLOWeightedDataset
# print("🚀 Clase YOLODataset interceptada con éxito. Usando YOLOWeightedDataset para balanceo.")

# Forzamos los parches de seguridad para evitar fallos de formato con el .yaml
os.environ["WANDB_PLOT_BBOX_ARR"] = "false"
os.environ["WANDB_LOG_MODEL"] = "true"

# Forzar a Python a encontrar el módulo 'dinov2' y 'rad_dino'
DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)


#model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/ft_1.1/FT_100_default10/weights/best.pt")

#model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/lora_1.1/lora_100_default/weights/best.pt")

# model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/lp_1.1/linearprobing_100_default3/weights/best.pt")

model=YOLO("/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy/YOLO-DINO_pruebas_wandb/lp_base_50_medium_sgd_adapted30/weights/best.pt")

#model=YOLO("/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy/YOLO-DINO_pruebas_wandb/lp_base_50_medium_sgd_adapted37/weights/best.pt")

#37
# Ejecutar N iteraciones de bootstrap
N_ITER   = 1000
N_SAMPLE = 291
USE_CONTAINMENT = True



results = []

print(
    f"[DEBUG validation_prueba] USE_CONTAINMENT={USE_CONTAINMENT} "
    f"-> criterio TP: {'caja predicha contenida en la GT' if USE_CONTAINMENT else 'IoU estándar'}"
)
METRIC_PREFIX = "Pseudo-" if USE_CONTAINMENT else ""
METRIC_MODE = "containment" if USE_CONTAINMENT else "IoU"
if USE_CONTAINMENT:
    print(
        "[DEBUG validation_prueba] Las métricas mAP50 y mAP50-95 usan el mismo "
        "matching binario por containment; no representan umbrales IoU distintos."
    )

for i in range(N_ITER):

    # Forzar reset del validator cacheado
    if hasattr(model, 'validator'):
        del model.validator

    metrics = model.val(
        data='/mnt/nfs/home/dbenitom/pruebas/PTBRED-CISM-AP-MINO-EXTERNAL/ptbred-ap-mino.yaml',
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
        "iter"  : i,
        "mAP50" : metrics.box.map50,
        "mAP50-95"   : metrics.box.map,
        "Precision"     : metrics.box.mp,
        "Recall"     : metrics.box.mr,
    })
    print(f"[{i+1}/{N_ITER}] {METRIC_PREFIX}mAP({METRIC_MODE})={metrics.box.map50:.4f}")
    print(f"[{i+1}/{N_ITER}] {METRIC_PREFIX}mAP50-95({METRIC_MODE})={metrics.box.map:.4f}")
    print(f"[{i+1}/{N_ITER}] {METRIC_PREFIX}Precision({METRIC_MODE})={metrics.box.mp:.4f}")
    print(f"[{i+1}/{N_ITER}] {METRIC_PREFIX}Recall({METRIC_MODE})={metrics.box.mr:.4f}")

# Calcular intervalos de confianza al 95%
map50s = np.array([r["mAP50"] for r in results])
map_50_95s = np.array([r["mAP50-95"] for r in results])
precisions = np.array([r["Precision"] for r in results])
recalls = np.array([r["Recall"] for r in results])

# 2. Imprimir los resultados con su Media e Intervalo de Confianza al 95%
print("\n" + "=" * 55)
print("RESULTADOS FINALES DE BOOTSTRAP (IC 95%)")
print("=" * 55)

print(f"{METRIC_PREFIX}mAP({METRIC_MODE}) medio : {map50s.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}mAP({METRIC_MODE})    : [{np.percentile(map50s, 2.5):.4f}, {np.percentile(map50s, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}mAP50-95({METRIC_MODE}) medio : {map_50_95s.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}mAP50-95({METRIC_MODE}) : [{np.percentile(map_50_95s, 2.5):.4f}, {np.percentile(map_50_95s, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}Precision({METRIC_MODE}) media : {precisions.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}Precision({METRIC_MODE}): [{np.percentile(precisions, 2.5):.4f}, {np.percentile(precisions, 97.5):.4f}]\n")

print(f"{METRIC_PREFIX}Recall({METRIC_MODE}) medio    : {recalls.mean():.4f}")
print(f"IC 95% {METRIC_PREFIX}Recall({METRIC_MODE})   : [{np.percentile(recalls, 2.5):.4f}, {np.percentile(recalls, 97.5):.4f}]")
print("=" * 55)