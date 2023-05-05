# MONAI code base for Neptune Pathology Tissue Segmentation

## Training

```python
python main.py --data_dir='./dataset' --json_list='./dataset_NEPTUNE_CAPSULE.json' \
  --batch_size=64 --distributed --optim_lr=2e-3 --save_checkpoint \
  --logdir MONAI_NEPTUNE_CAPSULE_segresnet_stong_aug_lre-3_v5
```



## Catalog

