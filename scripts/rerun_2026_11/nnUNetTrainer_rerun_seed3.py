"""Rerun 03/11/2026 (PREREG_final_split1.md, experiment C): third seed replicate of the archived ResEnc-L lumen recipe.
Same class as nnUNetTrainer_250epochs_NoMirroring; a separate name gives the run its own results folder. seed2 lives in
nnUNetTrainer_rerun_stubs.py (02/11/2026)."""

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import (
    nnUNetTrainer_250epochs_NoMirroring,
)


class nnUNetTrainer_250epochs_NoMirroring_seed3(nnUNetTrainer_250epochs_NoMirroring):
    """Same recipe as the parent; separate name so the third ResEnc-L lumen run has its own folder."""
