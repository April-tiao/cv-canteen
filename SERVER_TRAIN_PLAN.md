# Server training plan

## 1. Source layout

New food source:

```text
/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/food_12000
```

Synthetic chart source:

```text
/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/合成图表
```

Other output:

```text
/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/网上其他
```

## 2. Build balanced other

Recommended target: 10k other.

The script has five source groups:

```text
natural:       CIFAR10/CIFAR100/OxfordPets/Flowers102
scene:         EuroSAT/DTD
document:      local RVL-CDIP/DocLayNet/PubLayNet/document screenshots
food_related:  local empty plates, menus, kitchens, cookware, packages
chart_related: local tables, dashboards, PPT, code, QR, maps, diagrams
```

If you only have public TorchVision data available, run:

```bash
python build_balanced_other.py \
  --output-dir images/网上其他 \
  --cache-dir public_dataset_cache \
  --target-count 10000 \
  --source-counts natural=5500,scene=4500,document=0,food_related=0,chart_related=0 \
  --overwrite
```

Better: add local hard negatives if you have them:

```bash
python build_balanced_other.py \
  --output-dir images/网上其他 \
  --cache-dir public_dataset_cache \
  --target-count 10000 \
  --source-counts natural=2500,scene=1500,document=2500,food_related=1500,chart_related=2000 \
  --document-root /PATH/TO/document_other \
  --food-related-root /PATH/TO/food_related_other \
  --chart-related-root /PATH/TO/chart_related_other \
  --overwrite
```

Local roots should preferably contain subfolders as labels, for example:

```text
document_other/
  form/
  ppt/
  paper/
  report/

food_related_other/
  empty_plate/
  menu/
  kitchen/
  cookware/

chart_related_other/
  excel_table/
  dashboard/
  qr_code/
  code/
  map/
```

## 3. Rebuild dataset

Use food_12000, existing images, synthetic charts, and balanced other.

First version: sample food to 10k, other to 10k.

```bash
python organize_dataset.py \
  --images-root images \
  --extra-food-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/food_12000 \
  --extra-chart-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/合成图表 \
  --extra-other-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/网上其他 \
  --output-root dataset \
  --max-food 10000 \
  --max-other 10000 \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite
```

Check counts:

```bash
find dataset/train/food -type f | wc -l
find dataset/train/chart -type f | wc -l
find dataset/train/other -type f | wc -l
find dataset/val/food -type f | wc -l
find dataset/val/chart -type f | wc -l
find dataset/val/other -type f | wc -l
```

## 4. Clean invalid images

```bash
python clean_invalid_images.py --root dataset --move
```

## 5. Training config

For this dataset, start with frozen backbone only:

```yaml
model:
  weights: IMAGENET1K_V1
  freeze_backbone_epochs: 8
  unfreeze_backbone: false

train:
  epochs: 8
  lr_head: 0.0003
  early_stopping_patience: 3

dataset:
  batch_class_counts:
    food: 20
    chart: 20
    other: 24
```

Run:

```bash
rm -rf checkpoints
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml --device cuda
```

Evaluate:

```bash
cat checkpoints/best_metrics.json
```

Focus on:

```text
food_recall
chart_recall
other_recall
other_to_food_fpr
other_to_chart_fpr
confusion_matrix
```
