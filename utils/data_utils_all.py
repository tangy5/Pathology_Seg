# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    TorchVisiond,
    Lambdad,
    Activationsd,
    OneOf,
    MedianSmoothd,
    AsDiscreted,
    Compose,
    CastToTyped,
    ComputeHoVerMapsd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    RandGaussianSmoothd,
    CenterSpatialCropd,
    AddChanneld,
    Resized,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    MapTransform,
)
from typing import Optional

from monai.config import KeysCollection
from monai.data import ImageReader, MetaTensor

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class AddNeptuneClassChanneld(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, value=1) -> None:
        super().__init__(keys, allow_missing_keys)
        self.value = value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            class_label = int(d["class"])

            # Create a new channel of zeros with the same shape as the image
            new_channel = torch.full((1, image.shape[1], image.shape[2]), class_label)

            # Stack the new channel to the image
            image_with_new_channel = torch.cat((image, new_channel), axis=0)
            d[key] = image_with_new_channel
        return d
    
def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            AddChanneld(keys=["label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                rotate_range=((np.pi), 0),
                scale_range=((0.2), (0.2)),
                shear_range=((0.05), (0.05)),
                translate_range=((6), (6)),
                padding_mode="zeros",
                mode=("nearest"),
            ),
            # CenterSpatialCropd(
            #     keys="image",
            #     roi_size=cfg["patch_size"],
            # ),
#             Resized(keys=["image", "label"], spatial_size=(1024, 1024)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandCoarseDropoutd(keys=["image"], prob=0.9, holes=50, spatial_size=20, fill_value=0),
            RandCoarseShuffled(keys=["image"], prob=0.5, holes=20, spatial_size=20),

            RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=0.9),

            RandAdjustContrastd(keys=["image"], prob=0.5),
#             MedianSmoothd(keys=["image"], radius=1),
            RandGaussianNoised(keys=["image"], prob=0.9, std=0.05),
#             OneOf(
#                 transforms=[
#                     RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
#                     MedianSmoothd(keys=["image"], radius=1),
#                     RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
#                 ]
#             ),
            CastToTyped(keys="image", dtype=np.uint8),
            TorchVisiond(
                keys=["image"],
                name="ColorJitter",
                brightness=(229 / 255.0, 281 / 255.0),
                contrast=(0.95, 1.10),
                saturation=(0.8, 1.2),
                hue=(-0.04, 0.04),
            ),
            # AsDiscreted(keys=["label_type"], to_onehot=[5]),
            AddNeptuneClassChanneld(keys=["image"]),

            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            # CastToTyped(keys="label_inst", dtype=torch.int),
            # AsDiscreted(keys=["label"], to_onehot=2),
            CastToTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    val_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            AddChanneld(keys=["label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
#             Resized(keys=["image", "label"], spatial_size=(1024, 1024)),

            CastToTyped(keys=["image", "label"], dtype=torch.int),
            # CenterSpatialCropd(
            #     keys="image",
            #     roi_size=cfg["patch_size"],
            # ),
            AddNeptuneClassChanneld(keys=["image"]),

            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            # CenterSpatialCropd(
            #     keys=["label", "hover_label_inst", "label_inst", "label_type"],
            #     roi_size=cfg["out_size"],
            # ),
            CastToTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    test_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
#             Resized(keys=["image", "label"], spatial_size=(1024, 1024)),

            CastToTyped(keys=["image", "label"], dtype=torch.int),
            # CenterSpatialCropd(
            #     keys="image",
            #     roi_size=cfg["patch_size"],
            # ),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            # CenterSpatialCropd(
            #     keys=["label", "hover_label_inst", "label_inst", "label_type"],
            #     roi_size=cfg["out_size"],
            # ),
            CastToTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
