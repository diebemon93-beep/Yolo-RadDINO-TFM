import pandas as pd

csv_path = "/mnt/nfs/projects/CXR-TB/DATA/data_cxr/public/vindr-cxr-png/annotations_test.csv"

df = pd.read_csv(csv_path)

target_classes = {
    "ILD",
    "Pulmonary fibrosis",
    "Other lesion",
    "Lung opacity"
}

# Clases presentes en cada imagen
classes_per_image = (
    df.groupby("image_id")["class_name"]
      .apply(set)
)

# Quedarnos solo con imágenes que tengan las 4 clases
matching_images = classes_per_image[
    classes_per_image.apply(lambda x: target_classes.issubset(x))
]

# Image IDs
image_ids = matching_images.index.tolist()

print(f"Imágenes que contienen las 4 clases: {len(image_ids)}")

for image_id in image_ids:
    print(image_id)

# Guardar
pd.DataFrame({"image_id": image_ids}).to_csv(
    "images_with_all_4_classes.csv",
    index=False
)