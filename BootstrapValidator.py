import numpy as np
from torch.utils.data import DataLoader, Sampler
from ultralytics.models.yolo.detect import DetectionValidator


class BootstrapSampler(Sampler):
    """Sampler de Bootstrap para evaluación con reemplazo.
    Usa un generador con semilla base fija para que la secuencia de muestreos 
    sea 100% idéntica entre distintas ejecuciones/modelos.
    """
    # 1. Fijamos una semilla base inicial fija para toda la clase
    _MASTER_SEED = 42  
    # 2. Generador de secuencias de semillas
    _seed_generator = np.random.default_rng(_MASTER_SEED)

    @classmethod
    def reset_sequence(cls, seed: int = 42):
        """Reinicia la secuencia de semillas si quieres volver a empezar desde 0 en el mismo script."""
        cls._MASTER_SEED = seed
        cls._seed_generator = np.random.default_rng(seed)

    def __init__(self, n_total: int, n_sample: int):
        self.n_total = n_total
        self.n_sample = n_sample
        # 3. Extrae la SIGUIENTE semilla de la secuencia (cada instancia recibe una distinta)
        # Usamos .integers() para obtener un entero válido como semilla
        self.seed = int(BootstrapSampler._seed_generator.integers(0, 2**31 - 1))

    def __iter__(self):
        # Mantiene la semilla asignada a esta iteración
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(self.n_total, size=self.n_sample, replace=True)

        # --- COMPROBACIÓN DE DUPLICADOS ---
        uniques = np.unique(indices)
        n_uniques = len(uniques)
        n_duplicados = self.n_sample - n_uniques
        pct_unicos = (n_uniques / self.n_sample) * 100

        print(
            f"\n[BootstrapSampler] Muestreo (Seed actual={self.seed}):"
            f"\n  • Total muestreado: {self.n_sample}"
            f"\n  • Imágenes ÚNICAS seleccionadas: {n_uniques} ({pct_unicos:.2f}%)"
            f"\n  • Repeticiones/Duplicados: {n_duplicados}"
            f"\n  • Primeros 10 índices: {indices[:10].tolist()}\n"
        )
        # ----------------------------------

        return iter(indices.tolist())

    def __len__(self):
        return self.n_sample


class FastBootstrapValidator(DetectionValidator):

    def __init__(self, *args, n_bootstrap: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bootstrap = (
            n_bootstrap
            if n_bootstrap is not None
            else getattr(self.args, "n_bootstrap", 0)
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Build a dataloader with bootstrap sampling for validation."""
        use_containment = bool(getattr(self.args, "use_containment", False))
        print(
            f"[DEBUG FastBootstrapValidator] use_containment={use_containment} "
            f"-> criterio TP: {'containment (pred dentro de GT)' if use_containment else 'IoU estándar'}"
        )
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if getattr(dataset, "rect", False):
            dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
            dataset.rect = False
        n_total = len(dataset)
        n_sample = self.n_bootstrap or n_total

        # Se auto-asigna la semilla correspondiente en la secuencia
        sampler = BootstrapSampler(n_total=n_total, n_sample=n_sample)
        print(f"[FastBootstrapValidator] Building bootstrap dataloader: {n_sample}/{n_total} images")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=(
                self.device.type == "cuda" if hasattr(self, "device") else False
            ),
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def get_stats(self):
        """Return validation metrics after inference."""
        return super().get_stats()