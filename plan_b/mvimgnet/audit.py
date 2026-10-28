"""Inventory, pilot extraction and manifest helpers for MVImgNet2.0."""

from __future__ import annotations

import json
import os
import random
import tarfile
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from .colmap_io import read_sparse_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_category_map(path: str | Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(",", 1)
            out[key] = value.strip()
    return out


def archive_inventory(raw_root: str | Path) -> Dict:
    raw = Path(raw_root)
    archives = sorted(raw.glob("*.tar.gz"))
    partials = sorted(raw.glob("*.part"))
    invalid = sorted(raw.glob("*.invalid-html.*"))
    return {
        "raw_root": str(raw),
        "archives_present": len(archives),
        "archives_partial": len(partials),
        "invalid_html_files": len(invalid),
        "archives": [{"name": p.name, "bytes": p.stat().st_size} for p in archives],
        "partials": [{"name": p.name, "bytes": p.stat().st_size} for p in partials],
        "invalid_html": [{"name": p.name, "bytes": p.stat().st_size} for p in invalid],
    }


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise RuntimeError(f"Refusing archive path outside extraction root: {member_name}")
    return target


def extract_pilot_instances(
    archives: Iterable[str | Path],
    output_root: str | Path,
    max_instances: int = 12,
    min_images_hint: int = 6,
) -> List[str]:
    """Extract the first complete-looking instance prefixes from archive streams.

    This intentionally does not delete or mutate source archives.  It relies on
    MVImgNet tar ordering where each instance's sparse/images/masks entries are
    grouped together, and stops after the next prefix begins.
    """

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected: List[str] = []
    image_counts: Counter[str] = Counter()
    seen_new_after_full = False
    for archive in archives:
        archive = Path(archive)
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                name = member.name.strip("/")
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                prefix = "/".join(parts[:2])
                if prefix not in selected and len(selected) < max_instances:
                    selected.append(prefix)
                elif prefix not in selected and len(selected) >= max_instances:
                    seen_new_after_full = True
                if prefix in selected:
                    if member.isfile() and Path(name).suffix.lower() in IMAGE_EXTS and "/images/" in f"/{name}/":
                        image_counts[prefix] += 1
                    target = _safe_target(output, name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        src = tf.extractfile(member)
                        if src is None:
                            continue
                        with src, open(target, "wb") as dst:
                            dst.write(src.read())
                if seen_new_after_full:
                    return selected
    return selected


def _image_lookup(instance_path: Path) -> Dict[str, Path]:
    image_dir = instance_path / "images"
    out: Dict[str, Path] = {}
    if not image_dir.exists():
        return out
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.name] = p
            out[str(Path("images") / p.name)] = p
    return out


def _mask_path(instance_path: Path, image_name: str) -> Path | None:
    mask_dir = instance_path / "masks"
    if not mask_dir.exists():
        return None
    stem = Path(image_name).stem
    for ext in [Path(image_name).suffix, ".jpg", ".png", ".jpeg"]:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan_extracted_instances(extracted_roots: Iterable[str | Path], category_map: Dict[str, str], max_mask_checks: int = 200):
    rows: List[Dict] = []
    rejected: List[Dict] = []
    camera_models: Counter[str] = Counter()
    totals = Counter()
    mask_checked = 0
    mask_present = 0
    mask_invalid = 0
    for root in extracted_roots:
        root = Path(root)
        if not root.exists():
            continue
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            class_id = class_dir.name
            for instance_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
                instance_uid = f"{class_id}/{instance_dir.name}"
                image_lookup = _image_lookup(instance_dir)
                sparse = instance_dir / "sparse" / "0"
                reject = None
                if not image_lookup:
                    reject = "missing_images"
                elif not (sparse / "cameras.bin").exists() or not (sparse / "images.bin").exists():
                    reject = "missing_colmap_sparse"
                if reject:
                    rejected.append({"instance_uid": instance_uid, "reason": reject, "path": str(instance_dir)})
                    continue
                try:
                    cameras, images, _ = read_sparse_model(sparse, with_points=False)
                except Exception as exc:
                    rejected.append({"instance_uid": instance_uid, "reason": "colmap_parse_failed", "error": repr(exc), "path": str(instance_dir)})
                    continue
                valid_ids = []
                missing_paths = []
                mask_count = 0
                for image_id, rec in images.items():
                    if rec.camera_id not in cameras:
                        continue
                    image_path = image_lookup.get(rec.name) or image_lookup.get(Path(rec.name).name)
                    if image_path is None:
                        missing_paths.append(rec.name)
                        continue
                    valid_ids.append(image_id)
                    mask = _mask_path(instance_dir, rec.name)
                    if mask is not None:
                        mask_count += 1
                        mask_present += 1
                        if mask_checked < max_mask_checks:
                            mask_checked += 1
                            try:
                                with Image.open(mask) as m:
                                    m.verify()
                            except Exception:
                                mask_invalid += 1
                for cam in cameras.values():
                    camera_models[cam.model] += 1
                totals["rgb_files"] += len([k for k in image_lookup if "/" not in k])
                totals["registered_images"] += len(images)
                totals["missing_image_paths"] += len(missing_paths)
                if len(valid_ids) < 4:
                    rejected.append(
                        {
                            "instance_uid": instance_uid,
                            "reason": "too_few_valid_registered_views",
                            "valid_views": len(valid_ids),
                            "path": str(instance_dir),
                        }
                    )
                    continue
                rows.append(
                    {
                        "dataset": "mvimgnet2",
                        "class_id": class_id,
                        "class_name": category_map.get(class_id),
                        "instance_id": instance_dir.name,
                        "instance_uid": instance_uid,
                        "reconstruction_uid": "sparse/0",
                        "instance_path": str(instance_dir),
                        "num_rgb_files": len([k for k in image_lookup if "/" not in k]),
                        "num_registered_images": len(images),
                        "num_valid_views": len(valid_ids),
                        "valid_image_ids": sorted(valid_ids),
                        "missing_image_paths": missing_paths[:20],
                        "mask_files": mask_count,
                        "text": None,
                        "text_available": False,
                        "text_source": "missing",
                        "source": "pilot_extraction" if "extracted_pilot" in str(instance_dir) else "extracted",
                    }
                )
    view_counts = [r["num_valid_views"] for r in rows]
    view_stats = {}
    if view_counts:
        sorted_counts = sorted(view_counts)
        p95_index = min(len(sorted_counts) - 1, int(round(0.95 * (len(sorted_counts) - 1))))
        view_stats = {
            "min": min(view_counts),
            "median": float(median(view_counts)),
            "p95": sorted_counts[p95_index],
            "max": max(view_counts),
        }
    summary = {
        "classes_with_data": len({r["class_id"] for r in rows}),
        "instances_found": len(rows) + len(rejected),
        "instances_usable": len(rows),
        "totals": dict(totals),
        "views_per_usable_instance": view_stats,
        "camera_model_distribution": dict(camera_models),
        "mask_present_registered_views": mask_present,
        "mask_checked": mask_checked,
        "mask_invalid": mask_invalid,
        "caption_present": 0,
        "caption_missing": len(rows),
    }
    return rows, rejected, summary


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def split_instances(rows: List[Dict], seed: int = 0, ratios=(0.9, 0.05, 0.05)) -> List[Dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return []
    n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
    n_test = max(1, int(round(n * ratios[2]))) if n >= 3 else 0
    n_train = max(0, n - n_val - n_test)
    out = []
    for idx, row in enumerate(shuffled):
        split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
        assigned = dict(row)
        assigned["split"] = split
        assigned["split_source"] = "local_instance_split"
        assigned["split_seed"] = seed
        out.append(assigned)
    return out


def fixed_episode(row: Dict, seed: int = 0, k_context: int = 3) -> Dict:
    digest = hashlib.sha256(row["instance_uid"].encode("utf-8")).hexdigest()
    stable_offset = int(digest[:8], 16)
    rng = random.Random((seed + stable_offset) & 0xFFFFFFFF)
    ids = list(row["valid_image_ids"])
    ids = rng.sample(ids, k_context + 1)
    episode = {
        **row,
        "context_image_ids": ids[:k_context],
        "target_image_id": ids[k_context],
        "sample_seed": seed,
        "k_context": k_context,
        "episode_id": f"{row['instance_uid'].replace('/', '__')}__seed{seed}",
    }
    return episode


def write_audit_markdown(path: str | Path, report: Dict) -> None:
    path = Path(path)
    inv = report["inventory"]
    summary = report["summary"]
    lines = [
        "# MVImgNet2.0 Plan B Data Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Raw Archives",
        "",
        f"- archives_present: {inv['archives_present']}",
        f"- archives_partial: {inv['archives_partial']}",
        f"- invalid_html_files: {inv['invalid_html_files']}",
        f"- archive_bytes_total: {sum(a['bytes'] for a in inv['archives'])}",
        "",
        "## Extracted/Pilot Usable Data",
        "",
        f"- classes_with_data: {summary['classes_with_data']}",
        f"- instances_found_indexed: {summary['instances_found']}",
        f"- instances_usable: {summary['instances_usable']}",
        f"- RGB files indexed: {summary['totals'].get('rgb_files', 0)}",
        f"- registered images indexed: {summary['totals'].get('registered_images', 0)}",
        f"- missing COLMAP image paths: {summary['totals'].get('missing_image_paths', 0)}",
        f"- views per usable instance: {summary.get('views_per_usable_instance', {})}",
        f"- camera models: {summary.get('camera_model_distribution', {})}",
        "",
        "## Annotation Status",
        "",
        f"- object masks present on registered views: {summary['mask_present_registered_views']}",
        f"- mask files checked: {summary['mask_checked']}",
        f"- invalid masks in checked subset: {summary['mask_invalid']}",
        f"- captions/text present: {summary['caption_present']}",
        f"- captions/text missing: {summary['caption_missing']}",
        "",
        "Text policy: missing text is represented as text=None, text_available=False, text_source=missing.",
        "Object masks are stored only under supervision/diagnostics and are not model inputs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
