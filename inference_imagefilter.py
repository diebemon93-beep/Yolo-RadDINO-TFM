import pandas as pd

csv_path = "/mnt/nfs/projects/CXR-TB/DATA/data_cxr/public/vindr-pcxr-png/annotations_test.csv"

df = pd.read_csv(csv_path)

# Definir la cantidad EXACTA que deseas para tus clases objetivo
target_counts = {
    "Reticulonodular opacity": 2,
}

# 1. Contar la frecuencia de cada clase por imagen
class_counts = (
    df.groupby(["image_id", "class_name"])
      .size()
      .unstack(fill_value=0)
)

# 2. Asegurar que las columnas objetivo existan en la matriz
for cls in target_counts.keys():
    if cls not in class_counts.columns:
        class_counts[cls] = 0

# 3. Construir la condición única sobre las clases objetivo
condition = True
for cls, count in target_counts.items():
    condition &= (class_counts[cls] == count)

# 4. Filtrar imágenes
matching_images = class_counts[condition]
image_ids = matching_images.index.tolist()

print(f"Imágenes que contienen exactamente 2 'ILD' y 1 'Other lesion' (pudiendo tener más etiquetas distintas): {len(image_ids)}")

for image_id in image_ids:
    print(image_id)
