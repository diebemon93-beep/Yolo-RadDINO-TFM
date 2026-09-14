# -*- coding: utf-8 -*-
"""
verify_class_balance.py
-----------------------
Carga YOLOWeightedDataset sobre tus datos y compara la distribución de clases
en un único gráfico agrupado (Normal vs los 6 métodos de agregación estándar y avanzados).

Uso directo sin CLI:
    python verify_class_balance.py
"""

import sys
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CARGAR YOLOWeightedDataset DESDE RUTA EXTERNA
# =============================================================================

def load_weighted_dataset_class(dataset_path: str):
    """Importa YOLOWeightedDataset desde la ruta indicada."""
    spec   = importlib.util.spec_from_file_location("weighted_dataset", dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YOLOWeightedDataset


# =============================================================================
# FUNCIONES DE MUESTREO
# =============================================================================

def verify_class_balance(dataset, num_samples: int = 1000) -> Counter:
    """
    Muestrea `num_samples` imágenes según las probabilidades del dataset
    y cuenta las apariciones de cada clase.
    """
    all_labels = []
    num_samples = min(len(dataset.labels), num_samples)

    if dataset.train_mode:
        choices = np.random.choice(len(dataset.labels), size=num_samples,
                                   p=dataset.probabilities)
    else:
        choices = np.random.choice(len(dataset.labels), size=num_samples,
                                   replace=False)

    for i in choices:
        label = dataset.labels[i]["cls"]
        all_labels.extend(label.reshape(-1).astype(int))

    return Counter(all_labels)


# =============================================================================
# GRÁFICO COMPUESTO EN UNA SOLA FIGURA (6 MÉTODOS + ORIGINAL)
# =============================================================================

def plot_single_combined_graph(dataset_path_str: str, data_yaml: str,
                               agg_methods_dict: dict, num_samples: int,
                               imgsz: int, save_path=None):
    """
    Calcula las distribuciones de cada método del diccionario y las dibuja juntas en 
    un único gráfico de barras agrupadas junto al modo Normal.
    """
    WeightedDataset = load_weighted_dataset_class(dataset_path_str)

    # Configurar el patch temporal en Ultralytics
    import ultralytics.data.build as build
    build.YOLODataset = WeightedDataset

    from ultralytics.data.utils import check_det_dataset
    data_info = check_det_dataset(data_yaml)
    train_path = data_info["train"]
    class_names = list(data_info["names"].values())
    nc = len(class_names)

    # Diccionario para almacenar los conteos de cada configuración
    results = {}

    # 1. Obtener la distribución en Modo Normal (solo una vez)
    print("📊 Calculando distribución original (Normal mode)...")
    import weighted_dataset as wd_module
    wd_module.AGG_FUNC = np.mean  # Cualquier función sirve de placeholder para inicializar
    
    dataset = WeightedDataset(img_path=train_path, imgsz=imgsz, augment=False, data=data_info)
    dataset.train_mode = False
    results["Normal"] = verify_class_balance(dataset, num_samples)

    # 2. Obtener las distribuciones para cada método (estándar y avanzados)
    for agg_name, agg_func in agg_methods_dict.items():
        print(f"🔄 Calculando distribución para agg={agg_name}...")
        wd_module.AGG_FUNC = agg_func

        # Recargar para inyectar correctamente la función asignada
        WeightedDataset = load_weighted_dataset_class(dataset_path_str)
        build.YOLODataset = WeightedDataset

        dataset = WeightedDataset(img_path=train_path, imgsz=imgsz, augment=False, data=data_info)
        dataset.train_mode = True
        results[agg_name] = verify_class_balance(dataset, num_samples)

    # =============================================================================
    # RENDERIZADO DEL GRÁFICO COMBINADO
    # =============================================================================
    print("🎨 Generando gráfico combinado con funciones avanzadas...")
    
    # Modos a graficar (Normal primero, luego los del diccionario)
    modes_to_plot = ["Normal"] + list(agg_methods_dict.keys())
    
    # Paleta extendida de colores para los 7 elementos
    colors = {
        "Normal":    "#B0BEC5",  # Gris
        "mean":      "#2196F3",  # Azul
        "max":       "#4CAF50",  # Verde
        "sum":       "#FF9800",  # Naranja
        "median":    "#9C27B0",  # Púrpura
        "top2":      "#E91E63",  # Rosa / Magenta
        "logsum":    "#00BCD4",  # Cian
    }

    x_indexes = np.arange(nc)
    total_modes = len(modes_to_plot)
    
    # Ajustamos el ancho de barra para que entren 7 barras juntas por clase de forma limpia
    bar_width = 0.85 / total_modes 

    # Hacemos la figura un poco más ancha para evitar solapamientos por el incremento de barras
    fig, ax = plt.subplots(figsize=(max(16, nc * 1.5), 7))

    # Dibujar las barras una al lado de la otra desplazando el eje X
    for i, mode in enumerate(modes_to_plot):
        values = [results[mode].get(c, 0) for c in range(nc)]
        
        # Centrado geométrico exacto de las 7 barras en el índice x de la patología
        positions = x_indexes + (i * bar_width) - (0.85 / 2) + (bar_width / 2)
        
        ax.bar(positions, values, bar_width, 
               label=f"np.{mode}" if mode not in ["Normal", "top2", "logsum"] else mode, 
               color=colors.get(mode, "#333333"), 
               alpha=0.9 if mode == "Normal" else 0.8)

    # Formatear ejes y etiquetas
    ax.set_xlabel("Clases / Patologías (VinDr-CXR)", fontsize=11, fontweight="bold", labelpad=12)
    ax.set_ylabel("Frecuencia de Aparición (Counts)", fontsize=11, fontweight="bold")
    ax.set_title(f"Comparativa Global de Balanceo de Clases [{num_samples} muestras] — Métodos Estándar vs Avanzados", 
                 fontsize=14, fontweight="bold", pad=15)
    
    # Ajustar etiquetas de marcas de clase
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    
    ax.legend(fontsize=10, loc="upper right", frameon=True, shadow=True, ncol=2) # En dos columnas para que no tape datos
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    # Guardar o mostrar en servidor
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"🎉 ¡Éxito! Gráfico unificado de 7 variables guardado en: {save_path}")
    else:
        plt.show()


# =============================================================================
# CONFIGURACIÓN INTERNA Y EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    
    # ==========================================================================
    # ⚙️ CONFIGURACIÓN DE PARÁMETROS INTRA-SCRIPT
    # ==========================================================================
    DATA_YAML = "/mnt/nfs/home/dbenitom/pruebas/PTBRED-CISM-AP-MINO/ptbred-ap-mino.yaml"
    DATASET_PY = "/mnt/nfs/home/dbenitom/Yolo-DinoV2/weighted_dataset.py"
    SAMPLES = 10000
    IMGSZ = 640
    SAVE_PATH = "/mnt/nfs/home/dbenitom/Yolo-DinoV2/balance.png"
    
    # Diccionario completo que mapea el nombre con la función ejecutable real
    AGG_METHODS = {
        "mean":   np.mean,
        "max":    np.max,
        "sum":    np.sum,
        "median": np.median,
        "top2":   lambda x: np.mean(np.sort(x)[-2:]) if len(x) >= 2 else np.max(x),
        "logsum": lambda x: np.log(np.sum(np.exp(x)))
    }
    # ==========================================================================

    # Añadir el directorio de weighted_dataset.py al path del sistema
    sys.path.insert(0, str(Path(DATASET_PY).parent))

    plot_single_combined_graph(
        dataset_path_str=DATASET_PY,
        data_yaml=DATA_YAML,
        agg_methods_dict=AGG_METHODS,
        num_samples=SAMPLES,
        imgsz=IMGSZ,
        save_path=SAVE_PATH,
    )