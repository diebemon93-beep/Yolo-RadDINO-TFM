import os
import sys
import torch
from ultralytics import YOLO



# ==============================================================================
# 🔥 SEGURIDAD DDP: Pre-cargar DinoV2 en caché antes de que Ultralytics bifurque hilos
# ==============================================================================

# ==============================================================================

DINO_PATH = "/mnt/nfs/home/dbenitom/dinov2"
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)

if __name__ == "__main__":
    #model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/argumentos_base/ft/ft_base2/weights/best.pt") #FT
    #model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/argumentos_base/lora/lp_base/weights/best.pt") #lora
    model = YOLO("/mnt/nfs/home/dbenitom/Yolo-DinoV2-deterministic-copy/YOLO-DINO_pruebas_wandb/lp_base_50_medium_sgd_adapted33/weights/best.pt") #lp


    # Validación final 
    model.val(data="/mnt/nfs/home/dbenitom/pruebas/PTBRED-CISM-AP-MINO/ptbred-ap-mino.yaml",
        device=[7], 
        batch=32,
        visualize=True,
        save_txt=True,
        split='test'
        
    )