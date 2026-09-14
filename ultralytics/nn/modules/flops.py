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
    
    import torch
from thop import profile

import torch
from thop import profile
from ultralytics import YOLO

from torchinfo import summary
from ultralytics import YOLO

from torchinfo import summary
from ultralytics import YOLO

model = YOLO("/mnt/nfs/home/dbenitom/pruebas/vincxr/yolov8n_dinov2_b.yaml").model

# Reporte visual completo en consola
summary(
    model,
    input_size=(1, 3, 640, 640),
    col_names=[
        "input_size",      # Dimensiones de la entrada de cada capa
        "output_size",     # Dimensiones de la salida
        "num_params",      # Número de parámetros
        "mult_adds",       # MACs / FLOPs por capa
        "kernel_size",     # Tamaño del kernel (filtro) en Convoluciones/Pooling
        "params_percent"   # Porcentaje de parámetros que representa cada capa
    ],
    col_width=20,
    depth=3,               # Profundidad de subcapas a mostrar (ej. 1, 2, 3...)
    row_settings=["var_names"],
    verbose=0 # Muestra el nombre de las variables internas,
)


for parameter in model.model[0].parameters():
    parameter.requires_grad = False

stats = summary(model, input_size=(1, 3, 640, 640), verbose=0)

# Parámetros y Computación
macs = stats.total_mult_adds
params_totales = stats.total_params
params_entrenables = stats.trainable_params

# Tamaño de los pesos
memoria_pesos_mb = stats.total_param_bytes / (1024 ** 2)

print(f"MACs: {macs / 1e9:.3f} G")
print(f"Parámetros totales: {stats.total_params / 1e6:.2f} M")
print(f"Parámetros entrenables: {stats.trainable_params / 1e6:.2f} M")
print(f"Tamaño de los pesos: {memoria_pesos_mb:.2f} MB")
