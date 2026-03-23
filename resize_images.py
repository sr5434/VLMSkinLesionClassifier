#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def get_resample_filter() -> int:
    try:
        return Image.Resampling.BICUBIC
    except AttributeError:
        return Image.BICUBIC


def resize_images(root: Path, size: int, out_dir: Path | None, verbose: bool) -> int:
    resample = get_resample_filter()
    images = iter_images(root)
    processed = 0
    skipped = 0
    failed = 0

    for path in images:
        try:
            with Image.open(path) as img:
                if img.size == (size, size) and out_dir is None:
                    skipped += 1
                    if verbose:
                        print(f"skip  {path}")
                    continue

                resized = img.resize((size, size), resample=resample)

                target = path if out_dir is None else out_dir / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)

                save_kwargs = {}
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    if resized.mode not in {"RGB", "L"}:
                        resized = resized.convert("RGB")
                    save_kwargs["quality"] = 95

                resized.save(target, **save_kwargs)
                processed += 1

                if verbose:
                    print(f"wrote {target}")
        except Exception as exc:
            failed += 1
            print(f"error {path}: {exc}")

    print(
        "done "
        f"(processed={processed}, skipped={skipped}, failed={failed}, total={len(images)})"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all images under a directory to a fixed square size."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Root directory to search for images. Default: data",
    )
    parser.add_argument(
        "--size",
        default=256,
        type=int,
        help="Target size for width and height. Default: 256",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, images are overwritten in place.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file as it is processed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"data dir not found: {data_dir}")
        return 1
    return resize_images(data_dir, args.size, args.out_dir, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
