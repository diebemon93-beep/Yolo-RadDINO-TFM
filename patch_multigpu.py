# patch_multigpu.py
import os
import ultralytics

# 1. Encontrar de forma automática la ruta física de la librería instalada
ultralytics_path = os.path.dirname(ultralytics.__file__)
target_file = os.path.join(ultralytics_path, "utils", "__init__.py")

print(f"🔍 Buscando archivo objetivo en: {target_file}")

# 2. Código exacto del parche que se ejecutará en CADA GPU por separado
patch_code = '''
# ==============================================================================
# 🌟 PARCHE PARA BALANCEO EXCLUSIVO DE CLASES MINORITARIAS (MULTI-GPU COMPATIBLE)
# ==============================================================================
def _apply_weighted_dataset_patch():
    import numpy as np
    from ultralytics.data.dataset import YOLODataset
    import ultralytics.data.build as build

    class YOLOWeightedDataset(YOLODataset):
        def __init__(self, *args, mode="train", **kwargs):
            super(YOLOWeightedDataset, self).__init__(*args, **kwargs)
            self.train_mode = "train" in getattr(self, "prefix", "")
            
            self.count_instances()
            
            # 1. Identificar cuáles son las clases menos representadas (minoritarias)
            # Usamos la mediana: las clases con menos instancias que la mediana son declaradas "raras"
            umbral_minoritario = np.median(self.counts)
            self.minority_classes = np.where(self.counts <= umbral_minoritario)[0]
            
            # Pesos inversos estándar
            class_weights = np.sum(self.counts) / self.counts
            self.class_weights = np.array(class_weights)
            
            # Recomendado 'np.max' para que si hay una clase rara, domine el peso de la imagen
            self.agg_func = np.max 

            self.weights = self.calculate_weights()
            self.probabilities = self.calculate_probabilities()

        def count_instances(self):
            self.counts = [0 for _ in range(len(self.data["names"]))]
            for label in self.labels:
                cls = label['cls'].reshape(-1).astype(int)
                for id in cls:
                    self.counts[id] += 1
            self.counts = np.array(self.counts)
            self.counts = np.where(self.counts == 0, 1, self.counts)

        def calculate_weights(self):
            weights = []
            for label in self.labels:
                cls = label['cls'].reshape(-1).astype(int)
                
                if cls.size == 0:
                    weights.append(1.0)
                    continue
                
                # 2. Filtrar y quedarnos SOLO con las clases minoritarias presentes en ESTA imagen
                minority_cls_in_image = cls[np.isin(cls, self.minority_classes)]
                
                if minority_cls_in_image.size == 0:
                    # Si la imagen SOLO tiene clases muy comunes (ej. Cardiomegalia sola), 
                    # no se altera su frecuencia y se le asigna el peso base neutro
                    weights.append(1.0)
                else:
                    # Si contiene al menos una clase rara, calculamos el peso considerando
                    # únicamente las clases minoritarias de la imagen
                    weight = self.agg_func(self.class_weights[minority_cls_in_image])
                    weights.append(weight)
                    
            return weights

        def calculate_probabilities(self):
            total_weight = sum(self.weights)
            return [w / total_weight for w in self.weights]

        def __getitem__(self, index):
            if not self.train_mode:
                return self.transforms(self.get_image_and_label(index))
            else:
                sampled_index = np.random.choice(len(self.labels), p=self.probabilities)
                return self.transforms(self.get_image_and_label(sampled_index))

    build.YOLODataset = YOLOWeightedDataset
    print("🚀 [DDP GPU] YOLOWeightedDataset (Filtro Minoritario) inyectado exitosamente.")

# Ejecutar el parche de forma nativa al importar utilidades
_apply_weighted_dataset_patch()
# ==============================================================================
'''

# 3. Leer el archivo para comprobar si ya ha sido parcheado antes
with open(target_file, "r") as f:
    content = f.read()

if "YOLOWeightedDataset" in content:
    print("⚠️ El archivo ya tiene el parche aplicado. No se han hecho modificaciones.")
else:
    # Escribir el parche al final del archivo
    with open(target_file, "a") as f:
        f.write(patch_code)
    print("🎉 ¡Parche inyectado con éxito en el core de Ultralytics! Listo para Multi-GPU.")