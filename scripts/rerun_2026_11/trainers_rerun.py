"""Rerun 02/11/2026: trainer classes of the Split-1 ensemble members, reconstructed from the archived
checkpoints because the original source files were not archived, plus the 1000-epoch pilot trainer.

Evidence (Split-1 checkpoint_final.pth, read with torch.load on the pod):
  nnUNetTrainer_ft500_NM : current_epoch 500, logging.lrs[0] = 1e-3, optimizer initial_lr 1e-3,
                           weight_decay 3e-5, Nesterov momentum 0.99, inference_allowed_mirroring_axes None,
                           mean_fg_dice at epoch 0 = 0.813 (started from the 250-epoch U-Net weights through
                           -pretrained_weights; nnU-Net re-initialises the .seg_layers heads).
  nnUNetTrainer_ft250_NM : current_epoch 250, lrs[0] = 1e-3, initial_lr 1e-3, mirroring None,
                           epoch-0 Dice 0.830 (started from the ResEnc-L checkpoint_final through
                           -pretrained_weights; scripts/training/finetune_resencl.sh: "250 epoch LR 1e-3").
Everything else is inherited from nnUNetTrainer_250epochs_NoMirroring (poly LR, SGD, no mirroring).
The same class names are used so that the archived Split-1 checkpoints load for inference as well.

nnUNetTrainer_1000epochs_NoMirroring (pilot, PREREG_pilot1000.md): the 250-epoch recipe with num_epochs = 1000,
i.e. the nnU-Net default schedule length without mirroring.
"""

import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import (
    nnUNetTrainer_250epochs_NoMirroring,
)


class nnUNetTrainer_ft500_NM(nnUNetTrainer_250epochs_NoMirroring):
    """U-Net member: 500 further epochs at initial LR 1e-3 from the 250-epoch U-Net (-pretrained_weights)."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.initial_lr = 1e-3


class nnUNetTrainer_ft250_NM(nnUNetTrainer_250epochs_NoMirroring):
    """ResEnc-L member: 250 further epochs at initial LR 1e-3 from the ResEnc-L final (-pretrained_weights)."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3


class nnUNetTrainer_1000epochs_NoMirroring(nnUNetTrainer_250epochs_NoMirroring):
    """Pilot (02/11/2026): same recipe as the 250-epoch trainer with the nnU-Net default length of 1000 epochs."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainer_250epochs_NoMirroring_seed2(nnUNetTrainer_250epochs_NoMirroring):
    """Same recipe as the parent; separate name so the second ResEnc-L lumen run (M2) has its own folder."""
