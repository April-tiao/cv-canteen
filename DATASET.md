# Dataset preparation

## Label rules

`organize_dataset.py` rebuilds the training dataset from `images/`.

Chart folders:

- `数据可视化图表_百度图片搜索`
- `条形图_百度图片搜索`
- `统计图表_百度图片搜索`
- `折线图_百度图片搜索`
- `柱状图_百度图片搜索`

Other folders:

- `other`
- `其他`
- `网上其他`
- `web_other`

All remaining top-level folders under `images/` are treated as `food`.

The script copies files into:

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

Original files under `images/` are not modified.

## Download food to 1000

Current local food count before web download was 427. On the server:

```bash
python download_web_images.py \
  --preset food \
  --output-dir images/网上食物 \
  --target-total 1000 \
  --existing-count 427 \
  --min-size 224
```

Then rebuild the dataset:

```bash
python organize_dataset.py \
  --images-root images \
  --output-root dataset \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite
```

## Prepare other

Yes, `other` data is needed for the `[0.0, 0.0]` target. Without it, the model does not learn the reject class and will tend to over-predict `food` or `chart`.

For a first version, download public generic negative images:

```bash
python download_web_images.py \
  --preset other \
  --output-dir images/网上其他 \
  --target-total 1500 \
  --existing-count 0 \
  --min-size 224
```

Better server-side `other` sources:

- Natural negatives: scenery, street, building, person, vehicle, indoor, furniture, clothes, electronics, plant, sports, tools, toys.
- Business negatives: app pages, chat screenshots, document pages, product images, package images, posters, logos, forms, report pages, PPT pages.
- Food-related negatives: empty plates, empty bowls, table, kitchen, cookware, restaurant environment, menu screenshots, takeaway bags, supermarket shelves, food illustrations.
- Chart-related negatives: Excel tables, normal table screenshots, dashboard pages, org charts, mind maps, maps, code screenshots, papers, financial report pages, forms, QR codes, barcodes, geometry diagrams, circuit diagrams, infographics, posters.

Put manually collected negatives into `images/网上其他` or `images/other`, then rerun `organize_dataset.py`.
