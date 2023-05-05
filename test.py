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

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR, UNet, SegResNet, UNETR
from segment_anything import build_sam, build_sam_vit_b
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./evaluation/segresnet/cpt", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="segresnet", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_NEPTUNE_test_capsule.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=256, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=256, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    # model = SwinUNETR(
    #     img_size=(args.roi_x, args.roi_y),
    #     in_channels=args.in_channels,
    #     out_channels=args.out_channels,
    #     feature_size=args.feature_size,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     dropout_path_rate=0.0,
    #     use_checkpoint=args.use_checkpoint,
    #     spatial_dims=2,
    # )
    
#     model = UNet(
#         spatial_dims=2,
#         in_channels=3,
#         out_channels=2,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#     )

    model = SegResNet(
        spatial_dims=2,
        init_filters=32,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        dropout_prob=0.2,
        blocks_down=[1,2,2,4],
        blocks_up=[1,1,1],
    )

#     model = UNETR(
#         in_channels=3,
#         out_channels=2,
#         img_size=(256, 256),
#         feature_size=16,
#         hidden_size=768,
#         mlp_dim=3072,
#         num_heads=12,
#         pos_embed="perceptron",
#         norm_name="instance",
#         res_block=True,
#         dropout_rate=0.0,
#     )
    
    # model = build_sam_vit_b(checkpoint=None)

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            # _, _, h, w, d = val_labels.shape
            # target_shape = (h, w, d)
            # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            # print("Inference on case {}".format(img_name))
            val_outputs = model(val_inputs)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()

            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)

            val_labels = val_labels.cpu().numpy()

            dice_list_sub = []
            for i in range(1, 2):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            if organ_Dice > 0.1:
                dice_list_case.append(mean_dice)
            # print(val_labels.shape)
            # img = (val_labels - val_labels.min())*255 / (val_labels.max() - val_labels.min())
            # plt.imsave(os.path.join(output_directory, str(i) + '_pred.png'),
            #             img)
            # # nib.save(
            #     nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            # )

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == "__main__":
    main()
