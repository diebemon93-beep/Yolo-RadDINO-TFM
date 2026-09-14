import sys

import os


import torch

# Forzar algoritmos deterministas en todo PyTorch
torch.use_deterministic_algorithms(True, warn_only=False)

os.environ['PYTHONHASHSEED'] = '1'


# Forzar a Python a buscar primero en tu carpeta local del proyecto
project_root = "/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy"  # Cambia esto a la ruta de tu proyecto
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 2. IMPORTANTE: Importamos las tareas de la instalación oficial
import ultralytics.nn.tasks as tasks


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

# os.environ["WANDB_RUN_ID"] = "u9gmq55e"  # Sustituye por el ID de tu Run original
# os.environ["WANDB_RESUME"] = "must"             # Obliga a W&B a conectar al Run existente

DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

if __name__ == "__main__":
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    lr0=0.01
    batch_size=32
    imgsz=518
    box= 13.5 # (float) box loss gain
    cls_= 0.8 # (float) cls loss gain (scale with pixels)
    dfl= 2.5
    

    if local_rank == 0:
        wandb.init(
            project="YOLO-DINO_pruebas_wandb",
            job_type="training",
            name="prueba_determ_sweep",
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

    # model = YOLO("/mnt/nfs/home/dbenitom/YOLO-DINO_pruebas_wandb/FT_100_default10/weights/last.pt")

    #/mnt/nfs/home/dbenitom/YOLO-DINO_pruebas_wandb/linearprobing_100_default3/weights/last.pt

    model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/yolov8n_dinov2_b.yaml")

# ==========================================
# VERSIÓN 1 (Configuración Leve / Mínima)
# ==========================================
    # model.train(
    #     data="/mnt/nfs/home/dbenitom/pruebas/vincxr/vincxr.yaml",
    #     epochs=50,
    #     batch=batch_size,
    #     device=[5],
    #     lr0=lr0,
    #     imgsz=imgsz,
    #     project="YOLO-DINO_pruebas_wandb",
    #     name="ft_v1_light_aug",
    #     fraction=1.0,
    #     dfl=dfl,
    #     cls=cls_,
    #     box=box,
    #     cos_lr=True,
    #     pretrained=False,
    #     optimizer='AdamW',
    #     freeze=1,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     degrees=0.0,
    #     translate=0.1,
    #     scale=0.1,
    #     shear=0.0,
    #     perspective=0.0,
    #     flipud=0.0,
    #     fliplr=0.1,
    #     bgr=0.0,
    #     mosaic=0.0,
    #     mixup=0.0,
    # )

# ==========================================
# VERSIÓN 2 (Configuración Moderada)
# ==========================================
    model.train(
        data="/mnt/nfs/home/dbenitom/pruebas/vincxr/vincxr.yaml",
        epochs=5,
        batch=batch_size,
        device=[4],
        lr0=lr0,
        imgsz=imgsz,
        project="YOLO-DINO_pruebas_wandb",
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

# ==========================================
# VERSIÓN 3 (Configuración Agresiva / Máxima)
# ==========================================
    # model.train(
    #     data="/mnt/nfs/home/dbenitom/pruebas/vincxr/vincxr.yaml",
    #     epochs=50,
    #     batch=batch_size,
    #     device=[6],
    #     lr0=lr0,
    #     imgsz=imgsz,
    #     project="YOLO-DINO_pruebas_wandb",
    #     name="ft_v3_heavy_aug",
    #     fraction=1.0,
    #     dfl=dfl,
    #     cls=cls_,
    #     box=box,
    #     cos_lr=True,
    #     pretrained=False,
    #     optimizer='AdamW',
    #     freeze=1,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.6,
    #     degrees=10.0,
    #     translate=0.3,
    #     scale=0.7,
    #     shear=3.0,
    #     perspective=0.0,
    #     flipud=0.0,
    #     fliplr=0.7,
    #     bgr=0.0,
    #     mosaic=1.0,
    #     mixup=0.2,
    # )

    model.val(
        data="/mnt/nfs/home/dbenitom/pruebas/PTBRED-CISM-LAT-MINO/ptbred-lat-mino.yaml",
        device=[6],
        batch=32,
    )

    if local_rank == 0:
        wandb.finish()