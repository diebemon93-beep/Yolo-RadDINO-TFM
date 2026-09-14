import sys

import os


import torch
# Forzar a Python a buscar primero en tu carpeta local del proyecto
project_root = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy"  # Cambia esto a la ruta de tu proyecto
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ultralytics.nn.tasks as tasks

# Cambia la estrategia de compartición para evitar saturar los File Descriptors
torch.multiprocessing.set_sharing_strategy('file_system')


# Forzar algoritmos deterministas en todo PyTorch
torch.use_deterministic_algorithms(True, warn_only=False)

os.environ['PYTHONHASHSEED'] = '1'


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

# Valores por defecto del sweep
DEFAULTS = {
    "lr0":        1e-4,
    "box_weight": 7.5,
    "cls_weight": 0.5,
    "dfl_weight": 1.5,
}

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        wandb.init(
            project="YOLO-DINO_pruebas_wandb",
            job_type="training",
            config=DEFAULTS,
            mode="online",
        )
        config = dict(wandb.config)  # convertir a dict normal

        # Broadcast config a otros ranks via variable de entorno
        import json
        os.environ["SWEEP_CONFIG"] = json.dumps(config)
    else:
        # Ranks secundarios leen la config del proceso padre
        import json
        import time
        # Esperar a que rank 0 escriba la config
        for _ in range(30):
            if "SWEEP_CONFIG" in os.environ:
                break
            time.sleep(0.5)
        config = json.loads(os.environ.get("SWEEP_CONFIG", json.dumps(DEFAULTS)))

    if local_rank == 0:
        print(f"[Sweep] lr0={config['lr0']}, box={config['box_weight']}, "
              f"cls={config['cls_weight']}, dfl={config['dfl_weight']}")

    model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/yolov8n_dinov2_b.yaml")

    model.train(
        data="/mnt/nfs/home/dbenitom/pruebas/vincxr/vincxr.yaml",
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

    # model.val(
    #     data="/mnt/nfs/home/dbenitom/pruebas/vincxr/vincxr.yaml",
    #     device=None,
    #     batch=32,
    #     workers=8,
    # )

    if local_rank == 0 and wandb.run is not None:
            # Extraemos las métricas registradas en el summary por Ultralytics
            map50 = wandb.run.summary.get("metrics/mAP50(B)", 0.0)
            map50_95 = wandb.run.summary.get("metrics/mAP50-95(B)", 0.0)

            # Calculamos la métrica combinada
            custom_score = 0.1 * map50 + 0.9 * map50_95

            # Guardamos en el summary directamente para que el Sweep la lea sin problemas de steps
            wandb.run.summary["metrics/custom_score"] = custom_score