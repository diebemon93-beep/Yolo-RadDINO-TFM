import numpy as np
from torch.utils.data import DataLoader, Sampler
from ultralytics.models.yolo.detect import DetectionValidator


class BootstrapSampler(Sampler):
    """Bootstrap sampler for validation with replacement.
    It uses a fixed base seed generator so the sampling sequence stays consistent across runs and models.
    """
    # 1. Keep a fixed base seed for the entire class.
    _MASTER_SEED = 42
    # 2. Generator for seed sequences.
    _seed_generator = np.random.default_rng(_MASTER_SEED)

    @classmethod
    def reset_sequence(cls, seed: int = 42):
        """Reset the seed sequence to restart from 0 in the same script."""
        cls._MASTER_SEED = seed
        cls._seed_generator = np.random.default_rng(seed)

    def __init__(self, n_total: int, n_sample: int):
        self.n_total = n_total
        self.n_sample = n_sample
        # 3. Draw the next seed from the generator so each instance gets a different seed.
        self.seed = int(BootstrapSampler._seed_generator.integers(0, 2**31 - 1))

    def __iter__(self):
        # Keep the seed assigned to this iteration.
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(self.n_total, size=self.n_sample, replace=True)

        # --- DUPLICATE CHECK ---
        uniques = np.unique(indices)
        n_uniques = len(uniques)
        n_duplicates = self.n_sample - n_uniques
        pct_unique = (n_uniques / self.n_sample) * 100

        print(
            f"\nBootstrap sampling summary (seed={self.seed}):"
            f"\n  Total sampled: {self.n_sample}"
            f"\n  Unique images selected: {n_uniques} ({pct_unique:.2f}%)"
            f"\n  Repetitions/duplicates: {n_duplicates}"
            f"\n  First 10 indexes: {indices[:10].tolist()}\n"
        )

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
            f"use_containment={use_containment}; "
            f"matching criterion: {'containment (prediction inside GT)' if use_containment else 'standard IoU'}"
        )
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if getattr(dataset, "rect", False):
            dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
            dataset.rect = False
        n_total = len(dataset)
        n_sample = self.n_bootstrap or n_total

        # Auto-assign the next seed in the sequence.
        sampler = BootstrapSampler(n_total=n_total, n_sample=n_sample)
        print(f"Building bootstrap dataloader: {n_sample}/{n_total} images")

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