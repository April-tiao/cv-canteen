#!/usr/bin/env bash
set -euo pipefail

FOOD_ROOT="${FOOD_ROOT:-/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/food_12000}"
OTHER_EXTRA_ROOT="${OTHER_EXTRA_ROOT:-/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/other_extra}"
OTHER_WEB_ROOT="${OTHER_WEB_ROOT:-/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/网上其他}"

python organize_dataset.py \
  --images-root images \
  --extra-food-root "${FOOD_ROOT}" \
  --extra-other-root "${OTHER_WEB_ROOT}" \
  --extra-other-root "${OTHER_EXTRA_ROOT}" \
  --output-root dataset \
  --max-food 10000 \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite \
  --link hardlink

echo "dataset counts:"
find dataset/train/food -type f | wc -l | xargs echo "train food"
find dataset/train/chart -type f | wc -l | xargs echo "train chart"
find dataset/train/other -type f | wc -l | xargs echo "train other"
find dataset/val/food -type f | wc -l | xargs echo "val food"
find dataset/val/chart -type f | wc -l | xargs echo "val chart"
find dataset/val/other -type f | wc -l | xargs echo "val other"
