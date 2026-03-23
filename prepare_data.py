import argparse
import csv
import os
import random
import shutil
import sys


LABEL_MAP = {
    "benign": "0",
    "malignant": "1",
    "indeterminate/benign": "0",
    "indeterminate/malignant": "1",
}


def normalize_label(value):
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in LABEL_MAP:
        return LABEL_MAP[lowered]
    try:
        numeric = int(float(lowered))
    except ValueError:
        return None
    if numeric in (0, 1):
        return str(numeric)
    return None


def load_ids_by_class(metadata_path):
    ids_by_class = {"0": [], "1": []}
    skipped = {"missing_id": 0, "missing_label": 0, "unknown_label": 0}

    with open(metadata_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "isic_id" not in fieldnames:
            raise ValueError("Missing expected column in metadata: isic_id")
        if "target" in fieldnames:
            label_column = "target"
        elif "benign_malignant" in fieldnames:
            label_column = "benign_malignant"
        else:
            raise ValueError(
                "Missing expected columns in metadata: target or benign_malignant"
            )

        for row in reader:
            image_id = (row.get("isic_id") or "").strip()
            if not image_id:
                skipped["missing_id"] += 1
                continue
            raw_label = row.get(label_column)
            label = normalize_label(raw_label)
            if label is None:
                if raw_label is None or not str(raw_label).strip():
                    skipped["missing_label"] += 1
                else:
                    skipped["unknown_label"] += 1
                continue
            ids_by_class[label].append(image_id)

    return ids_by_class, skipped


def ensure_empty_dir(path, force):
    if not os.path.exists(path):
        return
    if force:
        shutil.rmtree(path)
        return
    if os.listdir(path):
        raise RuntimeError(
            f"Output directory {path!r} is not empty. Remove it or rerun with --force."
        )


def copy_split(ids, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    missing = 0
    for image_id in ids:
        filename = f"{image_id}.jpg"
        src = os.path.join(images_dir, filename)
        if not os.path.exists(src):
            missing += 1
            continue
        shutil.copy2(src, os.path.join(output_dir, filename))
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced train/test splits for skin lesion images."
    )
    parser.add_argument(
        "--metadata",
        default="archive/metadata.csv",
        help="Path to the metadata CSV.",
    )
    parser.add_argument(
        "--images",
        default="archive/images",
        help="Path to the image directory.",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory for train/test splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling/shuffling.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of images per class to put in train.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directory if it exists.",
    )
    args = parser.parse_args()

    ids_by_class, skipped = load_ids_by_class(args.metadata)
    if any(skipped.values()):
        print(
            "Note: skipped "
            f"{skipped['missing_id']} rows with missing isic_id, "
            f"{skipped['missing_label']} with missing label, "
            f"{skipped['unknown_label']} with unknown label."
        )
    count_benign = len(ids_by_class["0"])
    count_malignant = len(ids_by_class["1"])
    if count_benign == 0 or count_malignant == 0:
        raise RuntimeError(
            f"Not enough samples to balance (benign={count_benign}, malignant={count_malignant})."
        )

    min_count = min(count_benign, count_malignant)
    rng = random.Random(args.seed)

    selected = {}
    for target, ids in ids_by_class.items():
        if len(ids) > min_count:
            selected_ids = rng.sample(ids, min_count)
        else:
            selected_ids = list(ids)
        rng.shuffle(selected_ids)
        selected[target] = selected_ids

    train_count = int(min_count * args.train_fraction)
    train_count = max(1, min(train_count, min_count - 1))

    label_names = {"0": "benign", "1": "malignant"}
    ensure_empty_dir(args.output, args.force)

    missing_total = 0
    for target, ids in selected.items():
        train_ids = ids[:train_count]
        test_ids = ids[train_count:]

        train_dir = os.path.join(args.output, "train", label_names[target])
        test_dir = os.path.join(args.output, "test", label_names[target])

        missing_total += copy_split(train_ids, args.images, train_dir)
        missing_total += copy_split(test_ids, args.images, test_dir)

    print(
        f"Prepared {args.output!r} with {train_count} train and {min_count - train_count} "
        f"test images per class."
    )
    if missing_total:
        print(f"Warning: {missing_total} images listed in metadata were not found.")


if __name__ == "__main__":
    main()
