# weighted_dataset.py
import numpy as np
from ultralytics.data.dataset import YOLODataset

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.
        """
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        # En Ultralytics, la variable self.prefix suele ser "train: " o "val: "
        self.train_mode = "train" in getattr(self, "prefix", "")

        # Contar instancias por clase para balanceo inverso
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Función de agregación (puedes cambiar np.mean por np.max si prefieres dar prioridad a la clase rara)
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def count_instances(self):
        """Contar el número de instancias por clase en todo el dataset."""
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts) # Evitar división por cero

    def calculate_weights(self):
        """Calcular el peso agregado para cada imagen basado en sus cajas."""
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Peso por defecto si la imagen no contiene objetos (background)
            if cls.size == 0:
                weights.append(1.0)
                continue

            # Agregar los pesos usando la función seleccionada
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """Normalizar los pesos para convertirlos en probabilidades de muestreo."""
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """Retorna la imagen y etiquetas basándose en la probabilidad si está entrenando."""
        if not self.train_mode:
            # En validación se mantiene el flujo secuencial normal
            return self.transforms(self.get_image_and_label(index))
        else:
            # En entrenamiento, se ignora el 'index' secuencial y se realiza un muestreo probabilístico
            sampled_index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(sampled_index))