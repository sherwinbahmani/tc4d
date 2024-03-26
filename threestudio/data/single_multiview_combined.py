import math
import random
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.data.random_multiview import (
    RandomMultiviewCameraDataModuleConfig,
    RandomMultiviewCameraIterableDataset
)
from threestudio.utils.config import parse_structured
from threestudio.utils.base import Updateable
from threestudio.utils.typing import *

class RandomSingleMultiViewCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg_single_view: Any, cfg_multi_view: Any, multi_rate: int = None, multi_rate_perc: int = None) -> None:
        super().__init__()
        self.cfg_single = parse_structured(RandomCameraDataModuleConfig, cfg_single_view)
        self.cfg_multi = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg_multi_view)
        self.train_dataset_single = RandomCameraIterableDataset(self.cfg_single)
        self.train_dataset_multi = RandomMultiviewCameraIterableDataset(self.cfg_multi)
        self.idx = 0
        self.multi_rate = multi_rate
        self.multi_rate_perc = multi_rate_perc

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.train_dataset_single.update_step(epoch, global_step, on_load_weights)
        self.train_dataset_multi.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        if self.multi_rate is not None:
            multi = self.idx % self.multi_rate == 0
        elif self.multi_rate_perc is not None:
            multi = random.random() < self.multi_rate_perc
        else:
            multi = False
        if multi:
            batch = self.train_dataset_multi.collate(batch)
            batch['single_view'] = False
            batch['is_video'] = False
        else:
            batch = self.train_dataset_single.collate(batch)
            batch['single_view'] = True
        self.idx += 1
        return batch


@register("single-multiview-combined-camera-datamodule")
class SingleMultiviewCombinedCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_multi = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg.multi_view)
        self.cfg_single = parse_structured(RandomCameraDataModuleConfig, cfg.single_view)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomSingleMultiViewCameraIterableDataset(self.cfg_single, self.cfg_multi, multi_rate=self.cfg.multi_rate, multi_rate_perc=self.cfg.multi_rate_perc)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg_single, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg_single, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )