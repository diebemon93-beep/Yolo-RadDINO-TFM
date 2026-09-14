import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_global_metrics(file_path):
    """Lee las iteraciones Bootstrap de un archivo .txt.

    Retorna un diccionario: {'mAP50': np.array([...]), ...}
    """
    metrics = {'mAP50': [], 'mAP50-95': [], 'Precision': [], 'Recall': []}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(
        r'\[\d+/\d+\]\s+(mAP50|mAP50-95|Precision|Recall)=([0-9.]+)'
    )
    for match in pattern.finditer(content):
        metric_name, value = match.group(1), float(match.group(2))
        metrics[metric_name].append(value)

    return {k: np.array(v) for k, v in metrics.items()}


def calculate_paired_bootstrap_percentiles(array_a, array_b, alpha=0.05):
    """Calcula la resta directa entre distribuciones Bootstrap ya existentes,

    extrae el IC 95% por el método de percentiles y cuenta los valores
    negativos.
    """
    min_len = min(len(array_a), len(array_b))
    if min_len == 0:
        return None

    # Cortar vectores si hubiera desajuste en el número de iteraciones
    arr_a_cut = array_a[:min_len]
    arr_b_cut = array_b[:min_len]

    # 1. Resta emparejada punto a punto (iteración a iteración del Bootstrap)
    diffs = arr_a_cut - arr_b_cut

    # 2. Estimación puntual (diferencia media)
    mean_diff = float(np.mean(diffs))

    # 3. Método de Percentiles sobre la distribución empírica de restas
    p_lower = float(np.percentile(diffs, 100 * (alpha / 2)))  # 2.5%
    p_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))  # 97.5%

    # 4. Prueba de hipótesis: ¿Contiene el 0 el intervalo?
    contains_zero = p_lower <= 0 <= p_upper
    significant = not contains_zero

    # 5. Conteo de diferencias negativas (A < B)
    neg_count = int(np.sum(diffs < 0))
    neg_percentage = float((neg_count / min_len) * 100)

    return {
        'mean_diff': mean_diff,
        'ci_percentile': (p_lower, p_upper),
        'contains_zero': contains_zero,
        'significant': significant,
        'samples': min_len,
        'neg_count': neg_count,
        'neg_percentage': neg_percentage,
        'arr_a': arr_a_cut,
        'arr_b': arr_b_cut,
        'diffs': diffs,
    }


def plot_metric_differences(metric_name, plot_data, output_dir):
    """Genera y guarda una figura con 3 subplots (uno por cada comparación)

    mostrando la distribución de las restas y el conteo de negativas.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    fig.suptitle(
        f'Distribución de las Diferencias Bootstrap - Métrica: {metric_name}',
        fontsize=16,
        fontweight='bold',
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (comp_label, stats) in enumerate(plot_data.items()):
        ax = axes[idx]
        diffs = stats['diffs']
        mean_d = stats['mean_diff']
        p_low, p_high = stats['ci_percentile']
        contains_zero = stats['contains_zero']
        neg_c = stats['neg_count']
        neg_pct = stats['neg_percentage']
        total_s = stats['samples']

        # Graficar Histograma + KDE
        sns.histplot(
            diffs,
            kde=True,
            ax=ax,
            color=colors[idx],
            bins=30,
            alpha=0.6,
            edgecolor='black',
        )

        # Línea de referencia en 0 (Línea roja)
        ax.axvline(
            0,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Sin Diferencia (0)',
        )

        # Línea de la Media
        ax.axvline(
            mean_d,
            color='black',
            linestyle='-',
            linewidth=2,
            label=f'Media: {mean_d:+.4f}',
        )

        # Líneas de los Percentiles (IC 95%)
        ax.axvline(
            p_low,
            color='darkblue',
            linestyle=':',
            linewidth=1.8,
            label=f'IC 95%: [{p_low:+.4f}, {p_high:+.4f}]',
        )
        ax.axvline(p_high, color='darkblue', linestyle=':', linewidth=1.8)

        # Título y etiquetas
        sig_text = (
            'NO SIGNIFICATIVA (Contiene 0)'
            if contains_zero
            else 'SIGNIFICATIVA (No contiene 0)'
        )
        ax.set_title(
            f'{comp_label}\n{sig_text}\nNegativas: {neg_c}/{total_s}'
            f' ({neg_pct:.1f}%)',
            fontsize=10,
            fontweight='bold',
            color='darkred' if contains_zero else 'darkgreen',
        )
        ax.set_xlabel(f'Diferencia en {metric_name}')
        ax.set_ylabel('Frecuencia' if idx == 0 else '')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Guardar figura
    fig_path = output_dir / f'dist_differences_{metric_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f' 📊 Gráfica guardada: {fig_path.name}')


def run_paired_bootstrap(files_dict, output_txt_path, debug_samples=10):
    """Genera el reporte comparativo analizando las restas de las distribuciones Bootstrap

    incluyendo el conteo de diferencias negativas y 4 gráficas asociadas.
    """
    data = {}
    print('📥 Cargando distribuciones Bootstrap desde los archivos...')
    for name, path in files_dict.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f'❌ No se encontró el archivo para {name}: {path}'
            )
        data[name] = parse_global_metrics(path)
        print(
            f"   ✓ {name}: {len(data[name]['mAP50'])} iteraciones Bootstrap"
            ' leídas.'
        )

    comparisons = [('LoRA', 'FT'), ('LoRA', 'LP'), ('FT', 'LP')]
    metrics_list = ['mAP50', 'mAP50-95', 'Precision', 'Recall']

    output_path = Path(output_txt_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(
            '=======================================================================\n'
        )
        f.write(
            '   ANÁLISIS DE DIFERENCIAS PAREADAS (MÉTODO DE PERCENTILES'
            ' BOOTSTRAP)\n'
        )
        f.write(
            '=======================================================================\n\n'
        )

        for metric in metrics_list:
            f.write(
                '=======================================================================\n'
            )
            f.write(f'=== MÉTRICA: {metric}\n')
            f.write(
                '=======================================================================\n\n'
            )

            # -----------------------------------------------------------------
            # 1. VALORES INDIVIDUALES
            # -----------------------------------------------------------------
            f.write('--- METRICAS INDIVIDUALES (Percentiles 2.5 - 97.5) ---\n')
            for method_name in files_dict.keys():
                arr = data[method_name][metric]
                if len(arr) == 0:
                    f.write(
                        f'  • {method_name:<8} : Datos insuficientes.\n'
                    )
                    continue

                m = np.mean(arr)
                l_ci = np.percentile(arr, 2.5)
                u_ci = np.percentile(arr, 97.5)
                f.write(
                    f'  • {method_name:<8} (n={len(arr)}): Media = {m:.4f} | IC'
                    f' 95% = [{l_ci:.4f}, {u_ci:.4f}]\n'
                )

            f.write('\n')

            # -----------------------------------------------------------------
            # 2. COMPARACIONES EMPAREJADAS (Resta directa A - B)
            # -----------------------------------------------------------------
            f.write('--- COMPARACIONES EMPAREJADAS (A - B) ---\n')
            plot_data = {}

            for method_a, method_b in comparisons:
                arr_a = data[method_a][metric]
                arr_b = data[method_b][metric]

                stats_diff = calculate_paired_bootstrap_percentiles(
                    arr_a, arr_b
                )

                if stats_diff is None:
                    f.write(
                        f'  [{method_a} vs {method_b}]: Datos'
                        ' insuficientes.\n\n'
                    )
                    continue

                comp_label = f'{method_a} - {method_b}'
                plot_data[comp_label] = stats_diff

                mean_d = stats_diff['mean_diff']
                p_low, p_high = stats_diff['ci_percentile']
                contains_zero = stats_diff['contains_zero']
                neg_c = stats_diff['neg_count']
                neg_pct = stats_diff['neg_percentage']
                total_s = stats_diff['samples']

                status_str = (
                    'NO SIGNIFICATIVA (El IC 95% SÍ contiene al 0)'
                    if contains_zero
                    else 'SIGNIFICATIVA (El IC 95% NO contiene al 0)'
                )

                f.write(f'  ▸ Comparación: {method_a} - {method_b}\n')
                f.write(
                    '     • Iteraciones comparadas        :'
                    f' {total_s}\n'
                )
                f.write(f'     • Diferencia Media (A - B)      : {mean_d:+.4f}\n')
                f.write(
                    '     • IC 95% Percentil (Diferencia) :'
                    f' [{p_low:+.4f}, {p_high:+.4f}]\n'
                )
                f.write(
                    '     • ¿Contiene el cero?             :'
                    f' {"SÍ" if contains_zero else "NO"}\n'
                )
                f.write(
                    '     • Diferencias Negativas (A < B)  :'
                    f' {neg_c} / {total_s} ({neg_pct:.2f}%)\n'
                )
                f.write(f'     • Conclusión                    : {status_str}\n\n')

                # 🔍 DEBUG: Primeras restas punto a punto
                f.write(
                    f'     [DEBUG - Primeras {debug_samples} iteraciones'
                    f' {method_a} - {method_b}]:\n'
                )
                f.write(
                    f'       {"Iter":<6} | {method_a:<8} | {method_b:<8} | Resta'
                    f' ({method_a}-{method_b})\n'
                )
                f.write('       ' + '-' * 48 + '\n')

                n_show = min(debug_samples, total_s)
                for i in range(n_show):
                    val_a = stats_diff['arr_a'][i]
                    val_b = stats_diff['arr_b'][i]
                    diff_val = stats_diff['diffs'][i]
                    f.write(
                        f'       Iter {i+1:<2} | {val_a:<8.4f} | {val_b:<8.4f}'
                        f' | {diff_val:+07.4f}\n'
                    )

                f.write('\n' + '-' * 71 + '\n\n')

            # Generar gráfica de la métrica actual
            if plot_data:
                plot_metric_differences(metric, plot_data, output_dir)

    print('\n✅ Análisis de diferencias completado con éxito.')
    print(f'📄 Reporte guardado en:\n   👉 {output_path.resolve()}\n')


if __name__ == '__main__':
    FILES = {
        'LoRA': (
            '/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/lora_1.1_base_bootstrap.txt'
        ),
        'FT': (
            '/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/ft_1.1_base_bootstrap.txt'
        ),
        'LP': (
            '/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/lp_1.1_base_bootstrap.txt'
        ),
    }

    OUTPUT_FILE = '/mnt/nfs/home/dbenitom/pruebas/vincxr/history/definitivos/paired_bootstrap_results.txt'

    run_paired_bootstrap(FILES, OUTPUT_FILE, debug_samples=10)