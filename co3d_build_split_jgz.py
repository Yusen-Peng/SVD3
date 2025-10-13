#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# co3d_build_split_jgz.py

"""
Build strict train/test JGZ split files for CO3Dv2 categories.

Features / Fixes:
- Index by rec["image"]["path"] (ground truth key).
- Cameras read from rec["viewpoint"] (R, T, focal_length, principal_point).
- Exact category-prefixed relative paths (no jpg/png or subdir guessing).
- Optional: derive splits from set_lists (default) or meta.frame_splits.
- Optional: skip missing/near-black frames on disk.
- Optional: (NEW) omit invalid camera fields from records, or drop records
  entirely if any required camera field is invalid.
- Optional: allow camera fallback (identity/zeros) instead of failing.

Outputs:
  <root>/<category>_train.jgz
  <root>/<category>_test.jgz
"""

import argparse, os, os.path as osp, json, gzip, re, math
from glob import glob
from collections import defaultdict

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def write_json_gz(obj, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)

def canon(p: str) -> str:
    return re.sub(r"/+", "/", (p or "").replace("\\", "/"))

def guess_category(p: str) -> str:
    parts = canon(p).split("/")
    return parts[0] if parts else ""

def seq_from_rel(p: str) -> str:
    parts = canon(p).split("/")
    for i in range(len(parts) - 1):
        if parts[i+1] in ("images", "rgb", "color"):
            return parts[i]
    return parts[1] if len(parts) >= 3 else parts[0]

def load_setlists(cat_dir):
    """Merge all <cat_dir>/set_lists/set_lists_*.json into dict(train/val/test -> [rel_img_path])."""
    sl_dir = osp.join(cat_dir, "set_lists")
    files = sorted(glob(osp.join(sl_dir, "set_lists_*.json")))
    merged = {"train": [], "val": [], "test": []}
    for fp in files:
        try:
            data = read_json(fp)
        except Exception:
            continue
        for k in ("train", "val", "test"):
            items = data.get(k, [])
            for item in items:
                if isinstance(item, list) and len(item) >= 3:
                    merged[k].append(canon(item[2]))
                elif isinstance(item, str):
                    merged[k].append(canon(item))
    return merged

def normalize_frames_list(frames):
    """CO3D dumps can be list or dict of lists. Return a flat list of dicts."""
    if isinstance(frames, dict):
        flat = []
        for v in frames.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat
    return frames

def build_index_by_image_path(frames, category):
    """Index records by exact image.path with category prefix. No guessing."""
    by_path = {}
    for rec in frames:
        img = rec.get("image") or {}
        rel = canon(img.get("path", ""))
        if not rel:
            continue
        if not rel.startswith(category + "/"):
            continue  # different category; ignore
        by_path[rel] = rec
    return by_path

def get_cam_strict(rec, allow_fallback=False):
    vp = rec.get("viewpoint")
    if not isinstance(vp, dict):
        if allow_fallback:
            return [[1,0,0],[0,1,0],[0,0,1]], [0,0,0], [1.0,1.0], [0.0,0.0]
        raise KeyError("missing viewpoint")

    R  = vp.get("R")
    T  = vp.get("T")
    f  = vp.get("focal_length")
    p0 = vp.get("principal_point")
    if (R is None or T is None or f is None or p0 is None):
        if allow_fallback:
            R  = R  or [[1,0,0],[0,1,0],[0,0,1]]
            T  = T  or [0,0,0]
            f  = f  or [1.0, 1.0]
            p0 = p0 or [0.0, 0.0]
            return R, T, f, p0
        missing = [k for k,v in (("R",R),("T",T),("focal_length",f),("principal_point",p0)) if v is None]
        raise KeyError(f"viewpoint missing keys: {missing}")
    return R, T, f, p0

# ----------------------- validation helpers ----------------------
def _finite(x):
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    return False

def _shape(x, *dims):
    if not isinstance(x, (list, tuple)):
        return False
    if len(dims) == 1:
        return len(x) == dims[0]
    if len(x) != dims[0]:
        return False
    return all(isinstance(r, (list, tuple)) and len(r) == dims[1] for r in x)

def validate_viewpoint_fields(vp):
    """Return dict of only valid fields from viewpoint."""
    out = {}
    R  = vp.get("R");  T  = vp.get("T")
    f  = vp.get("focal_length"); p0 = vp.get("principal_point")
    if R  is not None and _shape(R,3,3) and _finite(R):  out["R"] = R
    if T  is not None and _shape(T,3)   and _finite(T):  out["T"] = T
    if f  is not None and _shape(f,2)   and _finite(f):  out["focal_length"] = f
    if p0 is not None and _shape(p0,2)  and _finite(p0): out["principal_point"] = p0
    return out

REQUIRED_KEYS = ("R","T","focal_length","principal_point")

def should_skip_image(abs_path, thresh=3):
    """Return True if file missing or looks near-black (grayscale mean<thresh)."""
    try:
        from PIL import Image
        import numpy as np
        if not osp.isfile(abs_path):
            return True
        im = Image.open(abs_path).convert("L")
        return float(np.asarray(im).mean()) < float(thresh)
    except Exception:
        return False  # don't skip if we couldn't check

def build_cat_splits(
    root, category, include_val_to=None, use_meta_splits=False,
    allow_camera_fallback=False, skip_black=False, black_thresh=3,
    omit_invalid_fields=False, drop_if_any_invalid=False
):
    """
    include_val_to: None | 'train' | 'test'
    use_meta_splits: if True, derive splits from rec['meta']['frame_splits'] instead of set_lists.
    omit_invalid_fields: keep record but omit invalid camera fields.
    drop_if_any_invalid: drop the record if any required camera field invalid or missing.
    """
    cat_dir = osp.join(root, category)
    fa_path = osp.join(cat_dir, "frame_annotations.jgz")
    if not osp.isfile(fa_path):
        return {}, {}

    frames = normalize_frames_list(read_json_gz(fa_path))
    by_path = build_index_by_image_path(frames, category)

    if use_meta_splits:
        split_by_path = {"train": set(), "test": set()}
        for rec in frames:
            img = rec.get("image", {})
            rel = canon(img.get("path", ""))
            if not rel.startswith(category + "/"):
                continue
            splits = (rec.get("meta", {}) or {}).get("frame_splits", [])
            is_train = any(("train" in s) for s in splits)
            (split_by_path["train"] if is_train else split_by_path["test"]).add(rel)
        train_paths = split_by_path["train"]
        test_paths  = split_by_path["test"]
    else:
        setlists = load_setlists(cat_dir)
        train_paths = set(setlists.get("train", []))
        test_paths  = set(setlists.get("test", []))
        val_paths   = set(setlists.get("val", []))
        if include_val_to == "train":
            train_paths |= val_paths
        elif include_val_to == "test":
            test_paths  |= val_paths
        train_paths = {p if p.startswith(category + "/") else f"{category}/{p}" for p in train_paths}
        test_paths  = {p if p.startswith(category + "/") else f"{category}/{p}" for p in test_paths}

    # --- assemble split dicts: seq -> list[records] ---
    def assemble(paths):
        out = defaultdict(list)
        missing = []
        skipped_black = 0
        dropped_invalid = 0
        for rel in sorted(paths):
            if guess_category(rel) != category:
                continue
            rel = canon(rel)
            rec = by_path.get(rel)
            if rec is None:
                missing.append(rel)
                continue

            if skip_black and should_skip_image(osp.join(root, rel), thresh=black_thresh):
                skipped_black += 1
                continue

            seq = str(rec.get("sequence_name") or seq_from_rel(rel))
            rec_out = {"filepath": rel}

            if omit_invalid_fields or drop_if_any_invalid:
                vp = rec.get("viewpoint") or {}
                valid = validate_viewpoint_fields(vp)
                if drop_if_any_invalid and not all(k in valid for k in REQUIRED_KEYS):
                    dropped_invalid += 1
                    continue
                # otherwise attach only valid fields (may be subset if omit_invalid_fields)
                rec_out.update(valid)
            else:
                # strict mode: either real cams or (if allowed) fallback
                try:
                    R, T, f, p0 = get_cam_strict(rec, allow_fallback=allow_camera_fallback)
                    rec_out.update({"R": R, "T": T, "focal_length": f, "principal_point": p0})
                except Exception:
                    dropped_invalid += 1
                    continue

            out[seq].append(rec_out)

        if missing:
            print(f"[{category}] MISSING in frame_annotations: {len(missing)}")
        if skip_black and (skipped_black):
            print(f"[{category}] SKIPPED near-black: {skipped_black}")
        if dropped_invalid:
            print(f"[{category}] DROPPED invalid: {dropped_invalid}")
        return out

    train_dict = assemble(train_paths)
    test_dict  = assemble(test_paths)
    return train_dict, test_dict

# ----------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CO3Dv2 root with category subfolders")
    ap.add_argument("--include_val_to", choices=["train", "test", "none"], default="test",
                    help="Fold 'val' frames into which split (default: test)")
    ap.add_argument("--use_meta_splits", action="store_true",
                    help="Build splits from meta.frame_splits instead of set_lists.")
    ap.add_argument("--allow_camera_fallback", action="store_true",
                    help="If a viewpoint field is missing, fallback to identity/zeros instead of raising.")
    ap.add_argument("--skip_black", action="store_true",
                    help="Skip frames whose image is missing or near-black.")
    ap.add_argument("--black_thresh", type=float, default=3.0,
                    help="Grayscale mean threshold for 'near-black' (0..255).")
    ap.add_argument("--omit_invalid_fields", action="store_true",
                    help="Keep records but drop invalid viewpoint fields.")
    ap.add_argument("--drop_if_any_invalid", action="store_true",
                    help="Drop a record if any required viewpoint field is invalid or missing.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cats = [d for d in os.listdir(args.root) if osp.isdir(osp.join(args.root, d))]
    total_train, total_test = 0, 0
    for cat in sorted(cats):
        cat_dir = osp.join(args.root, cat)
        fa_path = osp.join(cat_dir, "frame_annotations.jgz")
        sl_dir  = osp.join(cat_dir, "set_lists")
        if not osp.isfile(fa_path):
            continue
        if not args.use_meta_splits and not osp.isdir(sl_dir):
            # If no set_lists and not using meta, skip this category.
            continue

        print(f"[{cat}] building splits…")
        fold = None if args.include_val_to == "none" else args.include_val_to
        train_dict, test_dict = build_cat_splits(
            args.root, cat,
            include_val_to=fold,
            use_meta_splits=args.use_meta_splits,
            allow_camera_fallback=args.allow_camera_fallback,
            skip_black=args.skip_black, black_thresh=args.black_thresh,
            omit_invalid_fields=args.omit_invalid_fields,
            drop_if_any_invalid=args.drop_if_any_invalid
        )

        out_train = osp.join(args.root, f"{cat}_train.jgz")
        out_test  = osp.join(args.root, f"{cat}_test.jgz")

        if (not args.overwrite) and (osp.exists(out_train) or osp.exists(out_test)):
            print(f"[{cat}] outputs exist, use --overwrite to replace")
        else:
            write_json_gz(train_dict, out_train)
            write_json_gz(test_dict,  out_test)
            n_tr_frames = sum(len(v) for v in train_dict.values())
            n_te_frames = sum(len(v) for v in test_dict.values())
            print(f"[{cat}] wrote {out_train} (frames:{n_tr_frames}, seqs:{len(train_dict)})")
            print(f"[{cat}] wrote {out_test}  (frames:{n_te_frames}, seqs:{len(test_dict)})")
            total_train += len(train_dict)
            total_test  += len(test_dict)

    print(f"Done. Total sequences written  train={total_train}  test={total_test}")

if __name__ == "__main__":
    main()