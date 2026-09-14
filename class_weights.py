import numpy as np

# Nombres fijos de tus 8 clases médicas en orden de ID (0 a 7)
# CLASS_NAMES = [
#     "Bronchial thickening",
#     "Cardiomegaly",
#     "Consolidation",
#     "Diffuse alveolar opacity",
#     "Lung hyperinflation",
#     "Other opacity",
#     "Peribronchovascular opacity",
#     "Reticulonodular opacity",
# ]

import numpy as np

# Nombres fijos de tus 14 clases médicas en orden de ID (0 a 13)
CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

# ==============================================================================
# MODIFICA AQUÍ: Pon las instancias, frecuencias o porcentajes en orden del 0 al 13
# ==============================================================================
COUNTS_INPUT = [
    3098,  # Aortic enlargement (Clase base / máxima)
    187,   # Atelectasis
    458,   # Calcification
    2316,  # Cardiomegaly
    353,   # Consolidation
    397,   # ILD
    613,   # Infiltration
    1331,  # Lung Opacity
    841,   # Nodule/Mass
    1154,  # Other lesion
    1038,  # Pleural effusion
    2010,  # Pleural thickening
    96,    # Pneumothorax
    1621   # Pulmonary fibrosis
]

def main():
    print("=" * 70)
    print(" CALCULADORA DE PESOS POR FRECUENCIA INVERSA (MAX / VALUE)")
    print("=" * 70)
    print("Procesando la lista interna de conteos automáticamente...\n")
    
    # Validación de seguridad por si te falta o sobra algún número
    if len(COUNTS_INPUT) != len(CLASS_NAMES):
        print(f"[Error] La lista COUNTS_INPUT tiene {len(COUNTS_INPUT)} elementos, ")
        print(f"pero necesitas exactamente {len(CLASS_NAMES)} (uno para cada clase).")
        return

    class_counts = np.array(COUNTS_INPUT, dtype=float)
    
    # Lógica de frecuencia inversa respecto al valor máximo
    max_val = max(class_counts)
    class_weights = max_val / class_counts

    # Encontrar qué ID es la base (el máximo introducido)
    max_idx = np.argmax(class_counts)

    print("=" * 70)
    print(" OUTPUT LISTO PARA COPIAR Y PEGAR EN TU CONFIG/YAML:")
    print("=" * 70 + "\n")

    # Imprimir con el formato exacto requerido
    for i, (weight, name) in enumerate(zip(class_weights, CLASS_NAMES)):
        suffix = " (base)" if i == max_idx else ""
        print(f"    {i}: {weight:.4f},   # {name}{suffix}")
        
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()