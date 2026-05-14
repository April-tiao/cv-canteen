from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out-dir", default="configs/smoke")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for config_path in args.configs:
        path = Path(config_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["train"]["epochs"] = args.epochs
        data["train"]["early_stopping_patience"] = max(1, min(data["train"].get("early_stopping_patience", 1), args.epochs))
        out_path = out_dir / path.name
        out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
