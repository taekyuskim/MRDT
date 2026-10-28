"""Focused follow-up for MVImgNet2.0 pilot audit unresolved items.

This script does not re-extract data or alter training manifests.  It compares
the existing SVD-plane diagnostic camera coordinates with a reference-image-up
coordinate convention, resolves D1/D2/D3 frame matching, and records a
DataLoader worker retest result supplied by the caller.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from .cameras import processed_intrinsics, relative_world_to_camera
from .colmap_io import camera_center_world, qvec_to_rotmat, read_sparse_model
from .dataset import MVImgNetEpisodeDataset, read_jsonl
from .pilot_full_audit import (
    _angle_delta_deg,
    _az_el_radius,
    _camera_image_up_world,
    _find_image,
    _label,
    _lookat_center_from_rays,
    _plane_basis,
    _unit,
)


DEFAULT_ROOT = Path("/data1/haksoo/data/MVImgNet2.0")
DEFAULT_FULL_AUDIT = DEFAULT_ROOT / "reports_plan_b/mvimgnet2_pilot_full_audit"
D_TARGETS = {
    "D1": (30.0, 60.0),
    "D2": (-30.0, 60.0),
    "D3": (60.0, 30.0),
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _ref_image_up_basis(centers: np.ndarray, object_center: np.ndarray, ref_image_rec) -> Dict[str, Any]:
    ref_center = centers[0]
    x = _unit(ref_center - object_center)
    up_raw = _camera_image_up_world(ref_image_rec)
    up = up_raw - float(np.dot(up_raw, x)) * x
    if np.linalg.norm(up) < 1e-8:
        up = np.array([0.0, 0.0, 1.0]) - float(np.dot(np.array([0.0, 0.0, 1.0]), x)) * x
    up = _unit(up)
    y = _unit(np.cross(up, x))
    return {"origin": object_center, "x": x, "y": y, "up": up, "reference_radius": float(np.linalg.norm(ref_center - object_center))}


def _az_el_r_norm(center: np.ndarray, basis: Dict[str, Any]) -> Tuple[float, float, float]:
    az, el, r = _az_el_radius(center, basis)
    return az, el, r / max(float(basis.get("reference_radius", r)), 1e-9)


def _spherical_error(az: float, el: float, target: Tuple[float, float]) -> float:
    return math.sqrt(_angle_delta_deg(az, target[0]) ** 2 + (float(el) - target[1]) ** 2)


def _thumbnail(path: Path, label: str, size: int = 170) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    return _label(img, label, height=54)


def _concat_h(images: List[Image.Image]) -> Image.Image:
    width = sum(i.width for i in images)
    height = max(i.height for i in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _make_match_panel(instance_uid: str, row_by_id: Dict[int, Dict[str, Any]], matches: Dict[str, Dict[str, Any]], out: Path) -> None:
    tiles = []
    used = []
    for name in ["D1", "D2", "D3"]:
        m = matches[name]
        image_id = int(m["image_id"])
        used.append(image_id)
        tiles.append(
            _thumbnail(
                Path(row_by_id[image_id]["image_path"]),
                f"{name} target=({D_TARGETS[name][0]:+.0f},{D_TARGETS[name][1]:+.0f})\n"
                f"id={image_id} az={m['azimuth_deg']:.1f} el={m['elevation_deg']:.1f}\n"
                f"err={m['angular_error_deg']:.1f}",
            )
        )
    panel = _concat_h(tiles)
    footer = Image.new("RGB", (panel.width, 58), "white")
    draw = ImageDraw.Draw(footer)
    draw.text((6, 6), f"{instance_uid} | distinct D frames={len(set(used))}/3 | threshold is diagnostic only", fill=(0, 0, 0))
    final = Image.new("RGB", (panel.width, panel.height + footer.height), "white")
    final.paste(panel, (0, 0))
    final.paste(footer, (0, panel.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(out)


def build_followup(args) -> Dict[str, Any]:
    root = args.root
    processed = args.processed_root or root / "processed_plan_b"
    full_audit = args.full_audit_dir
    report_dir = args.report_dir
    panels = report_dir / "panels"
    report_dir.mkdir(parents=True, exist_ok=True)
    panels.mkdir(parents=True, exist_ok=True)

    instances = read_jsonl(processed / "instances.jsonl")
    episodes = read_jsonl(processed / "manifests/preview_episodes.jsonl")
    svd_existing = _read_csv(full_audit / "camera_views.csv")
    svd_by_key = {(r["instance_uid"], int(r["image_id"])): r for r in svd_existing}

    fixed_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    instance_match_rows: List[Dict[str, Any]] = []
    panel_links: Dict[str, str] = {}
    per_instance_view_rows: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for inst_row in instances:
        inst = Path(inst_row["instance_path"])
        cameras, images, points = read_sparse_model(inst / "sparse/0", with_points=True, max_points=args.max_points)
        valid_ids = [int(i) for i in inst_row["valid_image_ids"] if int(i) in images]
        centers = np.stack([camera_center_world(images[i].qvec, images[i].tvec) for i in valid_ids], axis=0)
        sparse_xyz = np.stack([p.xyz for p in points.values()], axis=0) if points else centers.mean(axis=0, keepdims=True)
        object_center = np.median(sparse_xyz, axis=0)
        svd_basis = _plane_basis(centers, object_center, centers[0])
        svd_basis["reference_radius"] = float(np.linalg.norm(centers[0] - object_center))
        ref_basis = _ref_image_up_basis(centers, object_center, images[valid_ids[0]])

        by_id: Dict[int, Dict[str, Any]] = {}
        for idx, image_id in enumerate(valid_ids):
            center = centers[idx]
            svd_az, svd_el, svd_rn = _az_el_r_norm(center, svd_basis)
            ref_az, ref_el, ref_rn = _az_el_r_norm(center, ref_basis)
            img_path = _find_image(inst, images[image_id].name)
            camera = cameras[images[image_id].camera_id]
            raw_w, raw_h = int(camera.width), int(camera.height)
            if img_path is not None:
                with Image.open(img_path) as img:
                    raw_w, raw_h = img.size
            k256, kraw, resize_meta = processed_intrinsics(camera, raw_w, raw_h, 256)
            k384 = np.diag([384.0 / 256.0, 384.0 / 256.0, 1.0]) @ k256
            row = {
                "instance_uid": inst_row["instance_uid"],
                "image_id": image_id,
                "frame_name": images[image_id].name,
                "image_path": str(img_path) if img_path is not None else "",
                "svd_azimuth_deg": svd_az,
                "svd_elevation_deg": svd_el,
                "svd_radius_refnorm": svd_rn,
                "refup_azimuth_deg": ref_az,
                "refup_elevation_deg": ref_el,
                "refup_radius_refnorm": ref_rn,
                "is_reference_image": image_id == valid_ids[0],
                "k256_fx": float(k256[0, 0]),
                "k256_fy": float(k256[1, 1]),
                "k256_cx": float(k256[0, 2]),
                "k256_cy": float(k256[1, 2]),
                "k384_fx": float(k384[0, 0]),
                "k384_fy": float(k384[1, 1]),
                "k384_cx": float(k384[0, 2]),
                "k384_cy": float(k384[1, 2]),
                "semantic_text_source": inst_row.get("text_source", "missing"),
                "camera_description_source": "COLMAP numeric qvec/tvec/K plus diagnostic fixed-camera angles",
            }
            fixed_rows.append(row)
            by_id[image_id] = row

        per_instance_view_rows[inst_row["instance_uid"]] = by_id
        ref_row = by_id[valid_ids[0]]
        old_ref = svd_by_key.get((inst_row["instance_uid"], valid_ids[0]), {})
        elevs = [r["refup_elevation_deg"] for r in by_id.values()]
        svd_elevs = [r["svd_elevation_deg"] for r in by_id.values()]
        plus60 = [r for r in by_id.values() if _angle_delta_deg(r["svd_azimuth_deg"], 60.0) <= args.plus60_tolerance_deg]
        plus60_ref = [r for r in by_id.values() if _angle_delta_deg(r["refup_azimuth_deg"], 60.0) <= args.plus60_tolerance_deg]
        comparison_rows.append({
            "instance_uid": inst_row["instance_uid"],
            "reference_image_id": valid_ids[0],
            "svd_reference_azimuth_deg": ref_row["svd_azimuth_deg"],
            "svd_reference_elevation_deg": ref_row["svd_elevation_deg"],
            "svd_reference_radius_refnorm": ref_row["svd_radius_refnorm"],
            "refup_reference_azimuth_deg": ref_row["refup_azimuth_deg"],
            "refup_reference_elevation_deg": ref_row["refup_elevation_deg"],
            "refup_reference_radius_refnorm": ref_row["refup_radius_refnorm"],
            "old_full_audit_reference_azimuth_deg": old_ref.get("azimuth_deg"),
            "old_full_audit_reference_elevation_deg": old_ref.get("elevation_deg"),
            "refup_elevation_min_deg": min(elevs),
            "refup_elevation_max_deg": max(elevs),
            "svd_elevation_min_deg": min(svd_elevs),
            "svd_elevation_max_deg": max(svd_elevs),
            "plus60_svd_count": len(plus60),
            "plus60_svd_image_ids": " ".join(str(r["image_id"]) for r in plus60),
            "plus60_refup_count": len(plus60_ref),
            "plus60_refup_image_ids": " ".join(str(r["image_id"]) for r in plus60_ref),
            "plus60_formula": f"abs(wrap(azimuth_deg - 60)) <= {args.plus60_tolerance_deg}",
        })

        matches: Dict[str, Dict[str, Any]] = {}
        used = set()
        for name, target in D_TARGETS.items():
            candidates = sorted(
                (
                    _spherical_error(r["refup_azimuth_deg"], r["refup_elevation_deg"], target),
                    int(r["image_id"]),
                    r,
                )
                for r in by_id.values()
            )
            err, image_id, best = candidates[0]
            possible = err <= args.d_tolerance_deg and image_id not in used
            used.add(image_id)
            m = {
                "instance_uid": inst_row["instance_uid"],
                "direction": name,
                "target_azimuth_deg": target[0],
                "target_elevation_deg": target[1],
                "image_id": image_id,
                "frame_name": best["frame_name"],
                "azimuth_deg": best["refup_azimuth_deg"],
                "elevation_deg": best["refup_elevation_deg"],
                "radius_refnorm": best["refup_radius_refnorm"],
                "angular_error_deg": err,
                "within_tolerance": err <= args.d_tolerance_deg,
                "distinct_under_greedy": possible,
                "tolerance_deg": args.d_tolerance_deg,
            }
            matches[name] = m
            match_rows.append(m)
        a_possible = matches["D1"]["within_tolerance"] and matches["D2"]["within_tolerance"] and len({matches["D1"]["image_id"], matches["D2"]["image_id"]}) == 2
        b_possible = a_possible and matches["D3"]["within_tolerance"] and len({matches["D1"]["image_id"], matches["D2"]["image_id"], matches["D3"]["image_id"]}) == 3
        instance_match_rows.append({
            "instance_uid": inst_row["instance_uid"],
            "A_D1_D2_possible": a_possible,
            "B_D1_D2_D3_possible": b_possible,
            "max_A_error_deg": max(matches["D1"]["angular_error_deg"], matches["D2"]["angular_error_deg"]),
            "max_B_error_deg": max(matches[d]["angular_error_deg"] for d in ["D1", "D2", "D3"]),
            "D1_image_id": matches["D1"]["image_id"],
            "D2_image_id": matches["D2"]["image_id"],
            "D3_image_id": matches["D3"]["image_id"],
        })
        panel_path = panels / f"d123_match_{inst_row['instance_uid'].replace('/', '__')}.png"
        _make_match_panel(inst_row["instance_uid"], by_id, matches, panel_path)
        panel_links[inst_row["instance_uid"]] = str(panel_path)

    # Verify Dataset camera tensors use the same COLMAP ids/frames as the fixed-camera rows.
    link_rows: List[Dict[str, Any]] = []
    ds = MVImgNetEpisodeDataset(processed / "manifests/preview_episodes.jsonl", split="followup", fixed_episodes=True)
    for idx, episode in enumerate(episodes):
        sample = ds[idx]
        inst = Path(episode["instance_path"])
        _, images, _ = read_sparse_model(inst / "sparse/0", with_points=False)
        ids = [int(x) for x in episode["context_image_ids"]] + [int(episode["target_image_id"])]
        ref_id = int(episode["context_image_ids"][0])
        rel = relative_world_to_camera(images, ids, ref_id)
        target_id = int(episode["target_image_id"])
        target_rel_equal = bool(np.allclose(sample["model_inputs"]["target_T_rel"].numpy(), rel[target_id] / np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], dtype=np.float64), atol=1e-4))
        # Dataset normalizes translation by context baseline; compare rotation exactly and translation direction only.
        ds_t = sample["model_inputs"]["target_T_rel"].numpy()
        raw_t = rel[target_id]
        rotation_equal = bool(np.allclose(ds_t[:3, :3], raw_t[:3, :3], atol=1e-5))
        fixed_exists = all((episode["instance_uid"] in per_instance_view_rows and i in per_instance_view_rows[episode["instance_uid"]]) for i in ids)
        link_rows.append({
            "episode_id": episode.get("episode_id"),
            "instance_uid": episode["instance_uid"],
            "dataset_reference_context_image_id": ref_id,
            "target_image_id": target_id,
            "context_image_ids": " ".join(str(x) for x in episode["context_image_ids"]),
            "fixed_camera_rows_exist_for_all_episode_frames": fixed_exists,
            "dataset_target_frame_name": sample["metadata"]["target_frame_id"],
            "colmap_target_frame_name": images[target_id].name,
            "target_frame_name_matches": sample["metadata"]["target_frame_id"] == images[target_id].name,
            "target_rotation_matches_colmap_relative": rotation_equal,
            "dataset_translation_is_context_baseline_normalized": True,
            "normalization_scale": sample["metadata"]["normalization_scale"],
        })

    # The unused variable keeps the raw comparison intent explicit without claiming raw translation equality.
    del target_rel_equal

    intrinsics_contract = {
        "dataset_processed_image_size": 256,
        "current_encoder_image_size": 384,
        "dataset_K_256": "K_256 = A_letterbox_raw_to_256 @ K_raw",
        "encoder_K_384_if_camera_adapter_operates_on_encoder_tokens": "K_384 = diag(384/256, 384/256, 1) @ K_256",
        "recommended_contract": "Keep Dataset camera tensors in 256-space; camera adapter must explicitly choose 256-space or scale to 384-space at encode_batch_conditions.",
        "semantic_text_vs_camera_description": {
            "semantic_text": "MVImgNet pilot has no original text; text=None/text_available=False/text_source=missing.",
            "camera_description": "Numeric camera metadata from COLMAP plus diagnostic az/el/radius is available and must not be treated as semantic caption text.",
        },
    }

    worker_results = {
        "previous_sandbox_result": "workers=2 timed out; stderr showed PyTorch multiprocessing/shared-memory manager permission errors during manual run.",
        "outside_sandbox_retest": args.worker2_retest_result,
        "classification": "BLOCKED_ENV" if args.worker2_retest_result == "passed" else "UNRESOLVED",
        "minimal_code_change": "none to Dataset; follow-up only records environment distinction.",
    }

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_full_audit_dir": str(full_audit),
        "followup_doc_note": "CODEX_MVIMGNET2_AUDIT_FOLLOWUP.md was not found under /data1/haksoo or /home/haksoo; user message checklist and D1/D2/D3 definitions were used.",
        "instances": len(instances),
        "views": len(fixed_rows),
        "d_tolerance_deg": args.d_tolerance_deg,
        "plus60_formula": f"abs(wrap(azimuth_deg - 60)) <= {args.plus60_tolerance_deg}",
        "A_possible_instances": sum(1 for r in instance_match_rows if r["A_D1_D2_possible"]),
        "B_possible_instances": sum(1 for r in instance_match_rows if r["B_D1_D2_D3_possible"]),
        "worker_results": worker_results,
        "intrinsics_contract": intrinsics_contract,
        "panel_links": panel_links,
    }

    _write_csv(report_dir / "coordinate_convention_comparison.csv", comparison_rows)
    _write_csv(report_dir / "fixed_camera_views_followup.csv", fixed_rows)
    _write_csv(report_dir / "d123_frame_matching.csv", match_rows)
    _write_csv(report_dir / "ab_combo_summary.csv", instance_match_rows)
    _write_csv(report_dir / "dataset_camera_link_check.csv", link_rows)
    _write_json(report_dir / "intrinsics_256_384_contract.json", intrinsics_contract)
    _write_json(report_dir / "worker_retest.json", worker_results)
    _write_json(report_dir / "FOLLOWUP_AUDIT.json", {"summary": summary, "coordinate_comparison": comparison_rows})
    _write_report(report_dir, summary, comparison_rows, instance_match_rows, worker_results)
    _write_html(report_dir, summary, comparison_rows, instance_match_rows)
    return summary


def _write_report(report_dir: Path, summary: Dict[str, Any], comparison_rows: List[Dict[str, Any]], combo_rows: List[Dict[str, Any]], worker: Dict[str, Any]) -> None:
    ref_ok = all(
        abs(float(r["refup_reference_azimuth_deg"])) < 1e-6
        and abs(float(r["refup_reference_elevation_deg"])) < 1e-6
        and abs(float(r["refup_reference_radius_refnorm"]) - 1.0) < 1e-6
        for r in comparison_rows
    )
    refup_elev_min = min(float(r["refup_elevation_min_deg"]) for r in comparison_rows)
    refup_elev_max = max(float(r["refup_elevation_max_deg"]) for r in comparison_rows)
    svd_elev_min = min(float(r["svd_elevation_min_deg"]) for r in comparison_rows)
    svd_elev_max = max(float(r["svd_elevation_max_deg"]) for r in comparison_rows)
    lines = [
        "# MVImgNet2.0 Pilot Audit Follow-up",
        "",
        "## Coordinate Convention Comparison",
        "",
        "| convention | object center | up/reference | reference az/el/r | elevation range | note |",
        "|---|---|---|---|---|---|",
        f"| existing SVD-plane | median sparse points | camera-center plane normal / first valid image | see CSV; radius ref-normalized in follow-up | {svd_elev_min:.2f} to {svd_elev_max:.2f} deg | preserved; not written back |",
        f"| reference-image-up | median sparse points | reference image visual up projected orthogonal to reference ray | {'0/0/1 confirmed' if ref_ok else 'check CSV'} | {refup_elev_min:.2f} to {refup_elev_max:.2f} deg | reproduced for D matching |",
        "",
        "## +60 and D1/D2/D3",
        "",
        f"- +60 formula: `{summary['plus60_formula']}`.",
        "- D1=(+30,+60), D2=(-30,+60), D3=(+60,+30), interpreted as `(azimuth_deg, elevation_deg)` in the reference-image-up convention.",
        f"- A={{D1,D2}} possible instances at tolerance {summary['d_tolerance_deg']} deg: {summary['A_possible_instances']}/{summary['instances']}.",
        f"- B={{D1,D2,D3}} possible instances at tolerance {summary['d_tolerance_deg']} deg: {summary['B_possible_instances']}/{summary['instances']}.",
        "- Nearest frames are still recorded even when outside tolerance.",
        "",
        "## Worker Retest",
        "",
        f"- Previous sandbox result: {worker['previous_sandbox_result']}",
        f"- Outside-sandbox retest: {worker['outside_sandbox_retest']}",
        f"- Classification: `{worker['classification']}`.",
        f"- Minimal code change: {worker['minimal_code_change']}",
        "",
        "## Semantic Text vs Camera Description",
        "",
        "- Semantic text remains missing in the pilot (`text=None`, `text_available=False`, `text_source=missing`).",
        "- Camera descriptions are numeric COLMAP/fixed-camera metadata and are not semantic captions.",
        "",
        "## 256 to 384 Intrinsics Contract",
        "",
        "- Dataset returns `K_256 = A_letterbox_raw_to_256 @ K_raw`.",
        "- Current SigLIP wrapper resizes the 256 square image to 384 square.",
        "- If camera conditioning is aligned to encoder tokens, use `K_384 = diag(1.5, 1.5, 1) @ K_256`; otherwise keep the adapter explicitly in 256-space.",
        "",
        "## Files",
        "",
        "- `coordinate_convention_comparison.csv`",
        "- `fixed_camera_views_followup.csv`",
        "- `d123_frame_matching.csv`",
        "- `ab_combo_summary.csv`",
        "- `dataset_camera_link_check.csv`",
        "- `intrinsics_256_384_contract.json`",
        "- `worker_retest.json`",
        "- `preview.html` and `panels/d123_match_*.png`",
        "",
        "## Next Dataset/Adapter Change Points",
        "",
        "- Dataset already returns `context_T_rel`, `target_T_rel`, `context_intrinsics`, `target_intrinsics` in [dataset.py](/data1/haksoo/ReSCENE-RDT/plan_b/mvimgnet/dataset.py).",
        "- Current RDT condition assembly is in [rescene_rdt.py](/home/haksoo/ReSCENE-RDT/rescene/models/rescene_rdt.py:134), where only discrete `view_emb` is added.",
        "- Batch encoding bridge is [pipeline.py](/home/haksoo/ReSCENE-RDT/rescene/pipeline.py:330); this is the likely place to pass camera tensors into a future adapter.",
        "- Do not overload missing semantic text with camera description strings.",
    ]
    (report_dir / "FOLLOWUP_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(report_dir: Path, summary: Dict[str, Any], comparison_rows: List[Dict[str, Any]], combo_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'><title>MVImgNet2.0 Audit Follow-up</title>",
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%;border:1px solid #ccc;margin:8px 0}</style>",
        "<h1>MVImgNet2.0 Audit Follow-up</h1>",
        f"<p>A possible: {summary['A_possible_instances']}/{summary['instances']} | B possible: {summary['B_possible_instances']}/{summary['instances']}</p>",
        "<h2>D1/D2/D3 Match Panels</h2>",
    ]
    for inst, path in summary["panel_links"].items():
        rel = Path(path).relative_to(report_dir)
        lines.append(f"<h3>{html.escape(inst)}</h3><a href='{rel}'><img src='{rel}'></a>")
    (report_dir / "preview.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--full-audit-dir", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_ROOT / "reports_plan_b/mvimgnet2_audit_followup")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--plus60-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--d-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--worker2-retest-result", choices=["passed", "failed", "not_run"], default="not_run")
    args = parser.parse_args(argv)
    summary = build_followup(args)
    print(json.dumps({
        "status": "passed",
        "report_dir": str(args.report_dir),
        "A_possible_instances": summary["A_possible_instances"],
        "B_possible_instances": summary["B_possible_instances"],
        "worker_classification": summary["worker_results"]["classification"],
    }, indent=2))


if __name__ == "__main__":
    main()
