# CV Canteen Classifier

Food / chart / other image classifier for canteen business images.

This project trains a ConvNeXt-Tiny based classifier with two sigmoid outputs:

```text
food_score
chart_score
```

It does not use a normal softmax 3-class head. The `other` class is represented by both outputs being negative.

## Task

Classify an input image into:

```text
food
chart
other
```

Label mapping:

```python
food  -> [1.0, 0.0]
chart -> [0.0, 1.0]
other -> [0.0, 0.0]
```

The model outputs:

```python
logits.shape == [batch_size, 2]
```

Training loss:

```python
torch.nn.BCEWithLogitsLoss()
```

Inference:

```python
food_score = sigmoid(logits[:, 0])
chart_score = sigmoid(logits[:, 1])

if food_score >= T_food and food_score >= chart_score:
    pred = "food"
elif chart_score >= T_chart and chart_score > food_score:
    pred = "chart"
else:
    pred = "other"
```

Default thresholds:

```text
T_food = 0.75
T_chart = 0.85
```

## Model

Backbone:

```text
TorchVision convnext_tiny
ImageNet1K pretrained weights
```

Classifier:

```text
Linear(in_features, 2)
```

The default training setup freezes the ConvNeXt backbone and trains only the final classifier head. This is intentional: the current task uses ordinary RGB images and the ImageNet pretrained visual features are useful enough for the first versions.

## Repository Layout

```text
.
├── model.py                       # ConvNeXt-Tiny model definition
├── dataset.py                     # Directory dataset and balanced batch sampler
├── transforms.py                  # Resize/pad/normalize and train augmentations
├── metrics.py                     # Accuracy, macro F1, recall, FPR, confusion matrix
├── train.py                       # Training entrypoint
├── threshold_search.py            # Threshold search without retraining
├── benchmark_infer.py             # Batch=1 latency benchmark
├── export_misclassified.py        # Export false predictions to CSV/images
├── organize_dataset.py            # Rebuild dataset from raw image folders
├── generate_synthetic_charts.py   # Generate synthetic chart images
├── build_balanced_other.py        # Build balanced other set from public datasets
├── prepare_dataset_server.sh      # Server dataset preparation helper
├── run_training.sh                # Single experiment launcher
├── run_ablation_queue.sh          # E0-E4 ablation queue
├── configs/                       # Experiment configs
└── README.md
```

Large data and artifacts are ignored by git:

```text
dataset/
images/
public_dataset_cache/
public_dataset_cache_local/
invalid_images/
checkpoints/
*.zip
*.tar.gz
```

## Dataset Format

Training expects:

```text
dataset/
  train/
    food/
    chart/
    other/
  val/
    food/
    chart/
    other/
```

Optional:

```text
dataset/
  test/
    food/
    chart/
    other/
```

Images are read from folder names and mapped to two-label targets.

## Raw Data Rules

`organize_dataset.py` rebuilds the standard dataset from `images/` and external roots.

Chart folders under `images/`:

```text
数据可视化图表_百度图片搜索
条形图_百度图片搜索
统计图表_百度图片搜索
折线图_百度图片搜索
柱状图_百度图片搜索
合成图表
```

Other folders under `images/`:

```text
other
其他
网上其他
web_other
other_extra
```

All remaining top-level folders under `images/` are treated as `food`.

## Preparing Dataset On Server

Typical server command:

```bash
bash prepare_dataset_server.sh
```

Equivalent explicit command:

```bash
python organize_dataset.py \
  --images-root images \
  --extra-food-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/food_12000 \
  --extra-other-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/网上其他 \
  --extra-other-root /HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/canteen2-511/images/other_extra \
  --output-root dataset \
  --max-food 10000 \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite \
  --link hardlink
```

Notes:

- `--max-food 10000` samples up to 10k food images.
- No `--max-other` means all available other images are used.
- `--link hardlink` avoids copying large numbers of files when possible.
- If hardlink fails, the script falls back to copying.

Check counts:

```bash
find dataset/train/food -type f | wc -l
find dataset/train/chart -type f | wc -l
find dataset/train/other -type f | wc -l
find dataset/val/food -type f | wc -l
find dataset/val/chart -type f | wc -l
find dataset/val/other -type f | wc -l
```

## Preprocessing

All splits use:

```text
RGB
keep-ratio resize
center padding to 224x224
ImageNet mean/std normalize
```

ImageNet normalize:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Train augmentation is label-aware:

- `food`: crop/flip/color/JPEG/blur/noise in default mode
- `chart`: light affine/color/JPEG/blur only
- `other`: light flip/color/JPEG

The E4 experiment uses light food augmentation:

```text
keep-ratio resize + padding
horizontal flip
light color jitter
light JPEG compression
normalize
```

This keeps food subjects intact and avoids making food too hard during training.

## Training

Default baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/head_only.yaml \
  --device cuda \
  --run-name "$(date +%Y%m%d_%H%M)_head_only"
```

Or use the helper:

```bash
bash run_training.sh configs/head_only.yaml head_only
```

The helper creates:

```text
checkpoints/YYYYMMDD_HHMM_method/
logs/YYYYMMDD_HHMM_method/train.log
logs/YYYYMMDD_HHMM_method/run_info.txt
```

For background training:

```bash
nohup bash run_training.sh configs/head_only.yaml head_only \
  > logs/head_only_nohup.log 2>&1 < /dev/null &
```

Monitor:

```bash
tail -f logs/*head_only*/train.log
nvidia-smi
ps -ef | grep train.py | grep -v grep
```

## Baseline Training Parameters

| Item | Baseline |
|---|---|
| Model | ConvNeXt-Tiny |
| Weights | ImageNet1K |
| Training | Freeze backbone, train classifier head only |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Image size | 224 |
| Batch sampler | food:20 / chart:20 / other:24 |
| Batch size | 64 |
| Epochs | 8 |
| Head LR | 0.0003 |
| Weight decay | 0.0001 |
| AMP | true |
| Grad clip | 1.0 |
| Early stopping | patience=3 |
| Workers | 2 |
| T_food | 0.75 |
| T_chart | 0.85 |

## Ablation Experiments

The planned experiments:

| ID | Purpose | Config |
|---|---|---|
| E0 | Threshold search, no retraining | `threshold_search.py` |
| E1 | Batch ratio food/chart/other = 28/18/18 | `configs/E1_head_food28_chart18_other18.yaml` |
| E2 | Batch ratio food/chart/other = 22/21/21 | `configs/E2_head_food22_chart21_other21.yaml` |
| E3 | Food positive weight = 1.5 | `configs/E3_head_food_pos_weight_1_5.yaml` |
| E4 | Light food augmentation | `configs/E4_head_light_food_aug.yaml` |

Run the full queue:

```bash
export BASE_CKPT=checkpoints/20260513_075638_head_only/best.pt
export GPU=0
export TIMEOUT_TRAIN=3h
export TIMEOUT_E0=2h

nohup bash run_ablation_queue.sh \
  > logs/ablation_queue_nohup.log 2>&1 < /dev/null &
```

The queue:

- runs experiments serially
- writes a separate log per experiment
- writes `.status` files
- continues to the next experiment if one fails or times out

Check queue:

```bash
tail -f logs/*_ablation_queue/queue.log
find logs -name "*.status" -type f -print -exec cat {} \;
```

## Threshold Search

Run threshold search on a checkpoint:

```bash
python -u threshold_search.py \
  --checkpoint checkpoints/YOUR_RUN/best.pt \
  --dataset-root dataset \
  --split val \
  --device cuda \
  --food-thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
  --chart-thresholds 0.80,0.85,0.90 \
  --max-other-to-food 0.02 \
  --max-other-to-chart 0.02 \
  --output-dir analysis/YYYYMMDD_HHMM_threshold_search
```

Outputs:

```text
threshold_search.csv
best_thresholds.json
```

## Evaluation Metrics

Training prints:

```text
accuracy
macro_f1
food precision / recall
chart precision / recall
other recall
other -> food false positive rate
other -> chart false positive rate
confusion matrix
```

Confusion matrix:

```text
rows = true label
cols = predicted label
[food, chart, other]
```

Important metrics:

```text
food_recall
chart_recall
other_recall
other_to_food_fpr
other_to_chart_fpr
```

## Export Misclassified Images

Example: export food images predicted as other.

```bash
python export_misclassified.py \
  --checkpoint checkpoints/YOUR_RUN/best.pt \
  --dataset-root dataset \
  --split val \
  --true-label food \
  --pred-label other \
  --output-csv analysis/food_as_other.csv \
  --copy-dir analysis/food_as_other_images \
  --device cuda
```

This is useful for checking whether `food` contains mislabeled menus, empty plates, packaging, or low-quality images.

## Synthetic Charts

Generate synthetic charts:

```bash
python generate_synthetic_charts.py \
  --output-dir images/合成图表 \
  --count 3000
```

Supported synthetic chart types:

```text
bar
line
pie
scatter
area
histogram
heatmap
```

## Inference Benchmark

Batch=1 latency benchmark:

```bash
python benchmark_infer.py \
  --checkpoint checkpoints/YOUR_RUN/best.pt \
  --image "$(find dataset/val/food -type f | head -n 1)" \
  --device cuda
```

Outputs:

```text
prediction
food_score
chart_score
mean_ms
p50_ms
p95_ms
p99_ms
```

## Known Practical Notes

1. Data loading can be the bottleneck.
   A800 speeds up model compute, not PIL decode or HDD/network file reads.

2. If `nvidia-smi` shows memory allocated but low GPU util, the GPU is probably waiting for data.

3. If DataLoader crashes with shared memory errors, reduce:

```yaml
num_workers: 2
```

4. Very large images are skipped during training using:

```yaml
max_image_pixels: 50000000
```

5. Runtime bad images are logged to:

```text
bad_images_runtime.log
```

## Recommended Workflow

1. Prepare dataset.
2. Run baseline head-only training.
3. Run threshold search.
4. Run E1-E4 ablations.
5. Compare `best_metrics.json`.
6. Pick the best checkpoint and threshold pair.
7. Export misclassified images.
8. Clean labels / add hard negatives if needed.
9. Retrain final model.

