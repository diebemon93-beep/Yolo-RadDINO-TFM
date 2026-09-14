import sys
import os

# 1. Rutas del proyecto e inyección de dependencias
project_root = "/mnt/nfs/home/dbenitom/Yolo-DinoV2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [CRUCIAL] Añade aquí la ruta a la carpeta que contiene el repositorio oficial 'dinov2' de Meta
# Por ejemplo, si lo clonaste en tu home: /mnt/nfs/home/dbenitom/dinov2
dinov2_meta_path = "/mnt/nfs/home/dbenitom/dinov2"  # <-- AJUSTA ESTA RUTA
if dinov2_meta_path not in sys.path:
    sys.path.insert(0, dinov2_meta_path)

# 2. Importamos las tareas de la instalación modificada
import ultralytics.nn.tasks as tasks

# 3. Importamos tu bloque customizado
from ultralytics.nn.modules.pretrained_vit import DinoV2Patches  

# 4. Inyección en los globales oficiales para el parser de Ultralytics
tasks.DinoV2Patches = DinoV2Patches
tasks.__dict__["DinoV2Patches"] = DinoV2Patches

# 5. Carga de librerías de ejecución
import wandb
from ultralytics import YOLO
from ultralytics.utils.callbacks import add_integration_callbacks

print("Cargando modelo personalizado...")
# Ahora torch.load podrá encontrar tanto 'dinov2' como 'DinoV2Patches'
# model = YOLO('/mnt/nfs/home/dbenitom/pruebas/vincxr/history/100epochas_linearprobing_augmentsfinales_v1/yolo-dino-exp635/weights/best.pt')
model = YOLO("yolov8n.pt")

print("Exportando modelo a formato ONNX...")
# Exportamos a ONNX. El opset=17 es ideal para soportar operadores complejos de ViT
model.export(format="onnx", opset=17, simplify=True)

print("¡Exportación completada con éxito!")