"""
multilabel_overlap_analysis.py
-------------------------------
Analiza la distribución de clases y el solapamiento multilabel en un dataset YOLO
para demostrar por qué el weighted sampling aumenta también las clases mayoritarias.

Ejecutar:
    python multilabel_overlap_analysis.py

Genera 4 figuras:
    1. Frecuencia de cada clase (barras)
    2. Matriz de co-ocurrencia (heatmap) — qué clases aparecen juntas
    3. Distribución de labels por imagen (cuántas clases por imagen)
    4. Para cada clase minoritaria: qué % de sus imágenes también contienen clases mayoritarias
"""

import sys
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml


# =============================================================================
# CONFIGURACIÓN ← EDITAR
# =============================================================================

DATA_YAML  = "/mnt/nfs/home/dbenitom/pruebas/PTBRED-CISM-AP-MAJO/ptbred-ap-majo.yaml"
LABELS_DIR = None   # None → se lee del yaml automáticamente
SAVE_DIR   = "/mnt/nfs/home/dbenitom/Yolo-DinoV2/"
N_SHOW     = 11      # clases minoritarias a analizar en detalle (las N más raras)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_dataset_info(data_yaml: str):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    base       = Path(data.get("path", "."))
    train      = data.get("train", "images/train")
    train_path = base / train if not Path(train).is_absolute() else Path(train)
    # Inferir labels desde images
    labels_path = Path(str(train_path).replace("images", "labels"))
    class_names = list(data["names"].values())
    return labels_path, class_names


def load_labels(labels_dir: Path, class_names: list):
    """
    Lee todos los .txt YOLO y devuelve:
      - image_classes: lista de sets, uno por imagen (clases presentes)
      - class_counts:  Counter con frecuencia de cada clase
    """
    nc = len(class_names)
    txt_files    = sorted(labels_dir.glob("*.txt"))
    image_classes = []
    class_counts  = Counter()

    for txt in txt_files:
        lines = txt.read_text().strip().splitlines()
        classes_in_img = set()
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                cls_id = int(parts[0])
                if 0 <= cls_id < nc:
                    classes_in_img.add(cls_id)
                    class_counts[cls_id] += 1
        image_classes.append(classes_in_img)

    return image_classes, class_counts, len(txt_files)


# =============================================================================
# ANÁLISIS
# =============================================================================

def compute_cooccurrence(image_classes: list, nc: int) -> np.ndarray:
    """Matriz de co-ocurrencia normalizada por frecuencia de la clase fila."""
    cooc = np.zeros((nc, nc), dtype=np.float32)
    for classes in image_classes:
        for a in classes:
            for b in classes:
                cooc[a, b] += 1
    # Normalizar: cooc[a,b] = P(b | a) = probabilidad de que b aparezca dado que a aparece
    row_sums = cooc.diagonal().copy()
    row_sums[row_sums == 0] = 1
    cooc_norm = cooc / row_sums[:, None]
    return cooc_norm


def compute_contamination(image_classes: list, class_counts: Counter,
                          nc: int, n_minority: int):
    """
    Para las N clases más raras, calcula qué porcentaje de sus imágenes
    también contienen cada clase mayoritaria.
    Retorna dict: {minority_cls: {other_cls: pct}}
    """
    sorted_by_freq = sorted(range(nc), key=lambda c: class_counts.get(c, 0))
    minority_cls   = sorted_by_freq[:n_minority]

    contamination = {}
    for mc in minority_cls:
        imgs_with_mc = [s for s in image_classes if mc in s]
        if not imgs_with_mc:
            continue
        co_counts = Counter()
        for s in imgs_with_mc:
            for other in s:
                if other != mc:
                    co_counts[other] += 1
        contamination[mc] = {
            other: co_counts[other] / len(imgs_with_mc)
            for other in range(nc) if other != mc
        }
    return contamination, minority_cls


# =============================================================================
# PLOTS
# =============================================================================

def plot_class_frequency(class_counts: Counter, class_names: list,
                         n_images: int, save_dir: str):
    nc     = len(class_names)
    counts = [class_counts.get(i, 0) for i in range(nc)]
    pcts   = [100 * c / n_images for c in counts]

    order  = np.argsort(counts)[::-1]
    names_sorted  = [class_names[i] for i in order]
    counts_sorted = [counts[i] for i in order]
    pcts_sorted   = [pcts[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(12, nc * 0.9), 5))
    bars = ax.bar(range(nc), counts_sorted, color="#2196F3", alpha=0.85, edgecolor="white")

    for bar, pct in zip(bars, pcts_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(range(nc))
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Nº de imágenes con la clase")
    ax.set_title(f"Frecuencia de clases  ({n_images} imágenes totales)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path(save_dir) / "1_class_frequency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


def plot_cooccurrence(cooc_norm: np.ndarray, class_names: list, save_dir: str):
    nc  = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, nc * 0.8), max(8, nc * 0.7)))

    im = ax.imshow(cooc_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="P(col | row)  — probabilidad de co-ocurrencia")

    ax.set_xticks(range(nc)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(nc)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Matriz de co-ocurrencia  P(clase columna | clase fila)\n"
                 "Valor alto = las dos clases aparecen juntas frecuentemente",
                 fontweight="bold")

    # Anotar valores altos
    for i in range(nc):
        for j in range(nc):
            if i != j and cooc_norm[i, j] > 0.3:
                ax.text(j, i, f"{cooc_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if cooc_norm[i, j] < 0.7 else "white")

    plt.tight_layout()
    out = Path(save_dir) / "2_cooccurrence_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


def plot_labels_per_image(image_classes: list, save_dir: str):
    n_labels = [len(s) for s in image_classes]
    counter  = Counter(n_labels)
    max_k    = max(counter.keys()) if counter else 0

    ks     = list(range(0, max_k + 1))
    counts = [counter.get(k, 0) for k in ks]
    pcts   = [100 * c / len(image_classes) for c in counts]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(ks, counts, color="#4CAF50", alpha=0.85, edgecolor="white")
    for bar, pct in zip(bars, pcts):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    mean_labels = np.mean(n_labels)
    ax.axvline(mean_labels, color="red", linestyle="--", linewidth=1.5,
               label=f"Media: {mean_labels:.2f} clases/imagen")

    ax.set_xlabel("Número de clases por imagen")
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Distribución de clases por imagen\n"
                 "(multilabel: una imagen puede tener varias clases)", fontweight="bold")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path(save_dir) / "3_labels_per_image.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


def plot_minority_contamination(contamination: dict, minority_cls: list,
                                class_names: list, class_counts: Counter,
                                n_images: int, save_dir: str):
    """
    Para cada clase minoritaria: muestra con qué frecuencia sus imágenes
    también contienen otras clases — demostrando el efecto multilabel en el sampling.
    """
    nc      = len(class_names)
    n_minor = len(minority_cls)
    if n_minor == 0:
        return

    ncols = min(n_minor, 4)
    nrows = (n_minor + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4),
                             tight_layout=True)
    axes_flat = np.array(axes).reshape(-1)

    for idx, mc in enumerate(minority_cls):
        ax   = axes_flat[idx]
        cont = contamination.get(mc, {})
        if not cont:
            ax.set_visible(False)
            continue

        other_ids   = [c for c in range(nc) if c != mc]
        other_names = [class_names[c] for c in other_ids]
        other_pcts  = [cont.get(c, 0) * 100 for c in other_ids]

        colors_bars = ["#F44336" if p > 50 else "#FF9800" if p > 20 else "#4CAF50"
                       for p in other_pcts]

        ax.barh(range(len(other_ids)), other_pcts, color=colors_bars, alpha=0.85)
        ax.set_yticks(range(len(other_ids)))
        ax.set_yticklabels(other_names, fontsize=7.5)
        ax.set_xlabel("% imágenes que también contienen esta clase", fontsize=8)
        ax.axvline(50, color="red", linestyle="--", linewidth=1, alpha=0.5)

        minority_pct = 100 * class_counts.get(mc, 0) / n_images
        ax.set_title(f"'{class_names[mc]}'\n"
                     f"({class_counts.get(mc,0)} imgs, {minority_pct:.1f}% del total)\n"
                     f"Rojo = >50% de sus imgs también tienen esa clase",
                     fontsize=8, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(minority_cls), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Contaminación multilabel: cuando se oversamplea una clase minoritaria,\n"
        "¿qué otras clases se arrastran? (barras rojas = co-ocurrencia >50%)",
        fontsize=11, fontweight="bold"
    )

    out = Path(save_dir) / "4_minority_contamination.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


def print_summary(contamination: dict, minority_cls: list,
                  class_names: list, class_counts: Counter, n_images: int):
    """Imprime un resumen textual del problema multilabel."""
    print("\n" + "="*65)
    print("RESUMEN: EFECTO MULTILABEL EN EL WEIGHTED SAMPLING")
    print("="*65)
    print(f"{'Clase minoritaria':<30} {'Freq':>6}  "
          f"{'Clase más co-ocurrente':<25} {'Co-oc%':>7}")
    print("-"*65)
    for mc in minority_cls:
        cont = contamination.get(mc, {})
        if not cont:
            continue
        top_other = max(cont, key=cont.get)
        top_pct   = cont[top_other] * 100
        freq      = class_counts.get(mc, 0)
        print(f"{class_names[mc]:<30} {freq:>6}  "
              f"{class_names[top_other]:<25} {top_pct:>6.1f}%")
    print("="*65)
    print("\n📌 CONCLUSIÓN:")
    print("   Al oversamplear una clase minoritaria, también se oversamplsa")
    print("   cualquier clase que co-ocurra en esas imágenes.")
    print("   → El weighted sampling en multilabel NO resuelve el desbalance")
    print("     de forma selectiva; aumenta TODO lo que aparece en esas imgs.")
    print("   → La alternativa correcta es cls_pw (loss weighting por clase).")
    print("="*65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n📂 Cargando dataset desde: {DATA_YAML}")
    labels_path, class_names = load_dataset_info(DATA_YAML)
    nc = len(class_names)

    if LABELS_DIR:
        labels_path = Path(LABELS_DIR)

    print(f"  Labels dir  : {labels_path}")
    print(f"  Clases      : {nc}  →  {class_names}")

    print("\n📖 Leyendo etiquetas...")
    image_classes, class_counts, n_images = load_labels(labels_path, class_names)
    print(f"  Imágenes leídas : {n_images}")
    print(f"  Con anotaciones : {sum(1 for s in image_classes if s)}")
    print(f"  Sin anotaciones : {sum(1 for s in image_classes if not s)}")

    print("\n🔢 Calculando co-ocurrencia...")
    cooc_norm = compute_cooccurrence(image_classes, nc)

    print(f"\n🔬 Analizando contaminación de las {N_SHOW} clases más raras...")
    contamination, minority_cls = compute_contamination(
        image_classes, class_counts, nc, N_SHOW
    )

    print_summary(contamination, minority_cls, class_names, class_counts, n_images)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n🎨 Generando figuras en: {SAVE_DIR}")

    plot_class_frequency(class_counts, class_names, n_images, SAVE_DIR)
    plot_cooccurrence(cooc_norm, class_names, SAVE_DIR)
    plot_labels_per_image(image_classes, SAVE_DIR)
    plot_minority_contamination(contamination, minority_cls, class_names,
                                class_counts, n_images, SAVE_DIR)

    print("\n✅ Análisis completo.")


if __name__ == "__main__":
    main()