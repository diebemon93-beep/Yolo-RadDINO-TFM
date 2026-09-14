import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(file_path):
    """
    Parses global metrics, precomputed CIs, and per-class metrics tables.
    """
    raw_global = {'mAP50': [], 'mAP50-95': [], 'Precision': [], 'Recall': []}
    ci_global = {}
    class_raw_data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. PARSE GLOBAL RUN METRICS ([347/1000] mAP50=0.1606)
    data_pattern = re.compile(r'\[\d+/\d+\]\s+(mAP50|mAP50-95|Precision|Recall)=([0-9.]+)')
    for match in data_pattern.finditer(content):
        metric_name, value = match.group(1), float(match.group(2))
        raw_global[metric_name].append(value)

    # 2. PARSE PRECOMPUTED GLOBAL CI95 BLOCK
    mean_pattern = re.compile(r'(mAP50|mAP50-95|Precision|Recall)\s+(?:medio|media)\s*:\s*([0-9.]+)', re.IGNORECASE)
    ci_pattern = re.compile(r'IC\s*95%\s*(mAP50|mAP50-95|Precision|Recall)\s*:\s*\[([0-9.]+),\s*([0-9.]+)\]', re.IGNORECASE)

    for match in mean_pattern.finditer(content):
        metric_name, mean_val = match.group(1), float(match.group(2))
        if metric_name not in ci_global: ci_global[metric_name] = {}
        ci_global[metric_name]['mean'] = mean_val

    for match in ci_pattern.finditer(content):
        metric_name, lower, upper = match.group(1), float(match.group(2)), float(match.group(3))
        if metric_name not in ci_global: ci_global[metric_name] = {}
        ci_global[metric_name]['ci'] = (lower, upper)

    # 3. PARSE PER-CLASS TABLES (Súper Robusto)
    # Busca directamente cualquier línea formateada como tabla de métricas de YOLO
    # Ej: "Aortic enlargement        211        212      0.103      0.467      0.141     0.0457"
    line_pattern = re.compile(
        r'^\s*([A-Za-z0-9/\s_-]+?)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',
        re.MULTILINE
    )

    for match in line_pattern.finditer(content):
        class_name = match.group(1).strip()
        
        # Ignoramos la fila 'all' ya que la gestionamos globalmente
        if class_name.lower() == 'all':
            continue

        p_val = float(match.group(4))
        r_val = float(match.group(5))
        map50_val = float(match.group(6))
        map50_95_val = float(match.group(7))

        if class_name not in class_raw_data:
            class_raw_data[class_name] = {'Precision': [], 'Recall': [], 'mAP50': [], 'mAP50-95': []}

        class_raw_data[class_name]['Precision'].append(p_val)
        class_raw_data[class_name]['Recall'].append(r_val)
        class_raw_data[class_name]['mAP50'].append(map50_val)
        class_raw_data[class_name]['mAP50-95'].append(map50_95_val)

    return {k: np.array(v) for k, v in raw_global.items()}, ci_global, class_raw_data


def calculate_percentile_ci95(data):
    """
    Calcula la media y el IC 95% usando percentiles directos (2.5% y 97.5%).
    """
    if len(data) == 0:
        return 0.0, (0.0, 0.0)
    
    mean_val = float(np.mean(data))
    lower_bound = float(np.percentile(data, 2.5))
    upper_bound = float(np.percentile(data, 97.5))
    
    return mean_val, (lower_bound, upper_bound)


def save_per_class_summary_txt(class_ci_results, output_txt_path):
    """Saves the calculated 95% CI for each class into a text file."""
    if not class_ci_results:
        print("⚠️ No se encontraron datos por clase para guardar en el TXT.")
        return

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("=======================================================\n")
        f.write("        PER-CLASS PERCENTILE RESULTS (IC 95%)          \n")
        f.write("=======================================================\n\n")
        
        for class_name, metrics in class_ci_results.items():
            f.write(f"--- Class: {class_name} ---\n")
            for metric_name, stats in metrics.items():
                mean = stats['mean']
                low, high = stats['ci']
                f.write(f"  {metric_name:<10} Mean: {mean:.4f} | 95% CI: [{low:.4f}, {high:.4f}]\n")
            f.write("\n")
            
    print(f"📄 Estadísticas IC95 por clase guardadas en:\n   👉 {output_txt_path}\n")


def plot_global_distribution(raw_global, ci_global, output_dir, file_stem):
    """Generates and saves the main global metrics image."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, metric_name in enumerate(['mAP50', 'mAP50-95', 'Precision', 'Recall']):
        ax = axes[idx]
        data = raw_global.get(metric_name, np.array([]))
        mean_val = ci_global.get(metric_name, {}).get('mean', np.mean(data) if len(data) else 0.0)
        ci_lower, ci_upper = ci_global.get(metric_name, {}).get('ci', (0.0, 0.0))

        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, color=colors[idx], bins=20, stat="density", alpha=0.4)

        ax.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(ci_lower, color='red', linestyle=':', linewidth=2, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='red', linestyle=':', linewidth=2)
        ax.axvspan(ci_lower, ci_upper, color='red', alpha=0.15)

        ax.set_title(f'Global Distribution of {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    global_img_path = output_dir / f"{file_stem}_global_metrics.png"
    plt.savefig(global_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Gráfica de métricas globales guardada en:\n   👉 {global_img_path}\n")


def plot_per_class_metric(class_raw_data, target_metric, output_dir, file_stem):
    """
    Genera 1 imagen con histogramas e IC 95% (percentiles directos 2.5% y 97.5%)
    para todas las clases individuales.
    """
    classes = list(class_raw_data.keys())
    num_classes = len(classes)
    
    if num_classes == 0:
        print(f"⚠️ No hay clases detectadas para procesar la métrica {target_metric}.")
        return None

    cols = 4
    rows = int(np.ceil(num_classes / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
    axes = axes.flatten() if num_classes > 1 else [axes]

    class_ci_summary = {}

    for idx, class_name in enumerate(classes):
        ax = axes[idx]
        data = np.array(class_raw_data[class_name][target_metric])
        
        # 🎯 Tu fórmula directa de percentiles
        mean_val, (ci_lower, ci_upper) = calculate_percentile_ci95(data)
        
        if class_name not in class_ci_summary:
            class_ci_summary[class_name] = {}
        class_ci_summary[class_name][target_metric] = {'mean': mean_val, 'ci': (ci_lower, ci_upper)}

        # Dibujar histograma
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, color='#2b5c8f', bins=15, stat="density", alpha=0.4)

        # Dibujar media e IC95
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
        ax.axvline(ci_lower, color='red', linestyle=':', linewidth=1.5, label=f'IC95: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='red', linestyle=':', linewidth=1.5)
        ax.axvspan(ci_lower, ci_upper, color='red', alpha=0.15)

        ax.set_title(class_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    # Ocultar subplots vacíos en la cuadrícula
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Class-level Distributions for {target_metric}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_img_path = output_dir / f"{file_stem}_class_distribution_{target_metric}.png"
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Gráfica por clase [{target_metric}] guardada en:\n   👉 {output_img_path}")

    return class_ci_summary


if __name__ == "__main__":
    LOG_FILE_PATH = "/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/argumentos_nodefault_error/ft_1.1/bootstrap.txt"
    
    log_path_obj = Path(LOG_FILE_PATH)
    output_dir = log_path_obj.parent
    file_stem = log_path_obj.stem

    # 1. Parse All Data
    raw_global, ci_global, class_raw_data = parse_log_file(LOG_FILE_PATH)

    print(f"\n✅ Total de clases detectadas en el log: {len(class_raw_data)}")
    if class_raw_data:
        print(f"   Clases: {', '.join(list(class_raw_data.keys()))}\n")

    # 2. Plot Global Metrics Image (1 Image)
    plot_global_distribution(raw_global, ci_global, output_dir, file_stem)

    # 3. Plot Per-Class Images (4 Images) & Calculate CIs
    all_class_cis = {}
    metrics_list = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    
    for metric in metrics_list:
        summary_metric = plot_per_class_metric(class_raw_data, metric, output_dir, file_stem)
        if summary_metric:
            for cls_name, stats in summary_metric.items():
                if cls_name not in all_class_cis:
                    all_class_cis[cls_name] = {}
                all_class_cis[cls_name].update(stats)

    # 4. Save Calculated Class CIs to a new TXT file
    output_txt_path = output_dir / f"{file_stem}_per_class_ci95.txt"
    save_per_class_summary_txt(all_class_cis, output_txt_path)