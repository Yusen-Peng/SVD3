#!/usr/bin/env python3
# co3d_build_split_jgz.py
import argparse, os, os.path as osp, json, gzip, re
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

def canon_path(p: str) -> str:
    # Normalize path separators and collapse multiple slashes
    p = p.replace("\\", "/")
    p = re.sub(r"/+", "/", p)
    return p

def guess_category_from_relpath(p: str) -> str:
    # "bench/415_.../images/frame000001.jpg" -> "bench"
    p = canon_path(p)
    parts = p.split("/")
    return parts[0] if parts else ""

def guess_seq_from_relpath(p: str) -> str:
    # "bench/415_.../images/frame000001.jpg" -> "415_..._..."
    p = canon_path(p)
    parts = p.split("/")
    # look for ".../<seq>/(images|rgb|color)/frame*.jpg"
    for i in range(len(parts) - 1):
        if parts[i+1] in ("images", "rgb", "color"):
            return parts[i]
    # fallback: middle component
    return parts[1] if len(parts) >= 3 else parts[0]

def load_setlists(cat_dir):
    """
    Merge all set_lists_*.json found under <cat_dir>/set_lists.
    Expected format: keys 'train', 'val', 'test' each maps to a list of
    [sequence_id, frame_index, rel_image_path].
    Returns: dict(split -> list of rel_image_path strings).
    """
    sl_dir = osp.join(cat_dir, "set_lists")
    files = sorted(glob(osp.join(sl_dir, "set_lists_*.json")))
    merged = {"train": [], "val": [], "test": []}
    for fp in files:
        try:
            data = read_json(fp)
        except Exception:
            continue
        for k in ("train", "val", "test"):
            if k in data and isinstance(data[k], list):
                for item in data[k]:
                    # accept [seq_id, frame_idx, path] or just "path"
                    if isinstance(item, list) and len(item) >= 3:
                        merged[k].append(canon_path(item[2]))
                    elif isinstance(item, str):
                        merged[k].append(canon_path(item))
    return merged

def build_cat_splits(root, category, include_val_to=None):
    """
    include_val_to: None | 'train' | 'test'
    Returns two dicts: train_dict, test_dict mapping seq_id -> [records...]
    """
    cat_dir = osp.join(root, category)
    fa_path = osp.join(cat_dir, "frame_annotations.jgz")
    if not osp.isfile(fa_path):
        return {}, {}

    # Load frame annotations and index by filepath
    frames = read_json_gz(fa_path)
    # frames can be a list OR dict — normalize to a flat list of records
    if isinstance(frames, dict):
        all_recs = []
        for v in frames.values():
            if isinstance(v, list):
                all_recs.extend(v)
        frames = all_recs

    by_path = {}
    for rec in frames:
        rel = rec.get("filepath") or rec.get("image_path") or rec.get("file_path")
        if not rel:
            # Try to reconstruct from pieces if present
            seq = (rec.get("sequence_id") or rec.get("seq_id")
                   or rec.get("sequence_name") or guess_seq_from_relpath(""))
            frm = rec.get("frame_number") or rec.get("frame_idx") or 0
            rel = f"{category}/{seq}/images/frame{int(frm):06d}.jpg"
        rel = canon_path(rel)
        # Ensure the rel path starts with the category (some dumps omit it)
        if not rel.startswith(category + "/"):
            rel = canon_path(f"{category}/{rel}")
        by_path[rel] = rec

    # Load and merge set-list files
    setlists = load_setlists(cat_dir)

    # Optionally fold 'val' into train or test
    train_paths = set(setlists.get("train", []))
    test_paths  = set(setlists.get("test", []))
    val_paths   = set(setlists.get("val", []))
    if include_val_to == "train":
        train_paths |= val_paths
        val_paths = set()
    elif include_val_to == "test":
        test_paths |= val_paths
        val_paths = set()

    # Helper to assemble split dicts: seq_id -> list[records]
    def make_split(paths):
        out = defaultdict(list)
        missing = 0
        for p in sorted(paths):
            # keep only paths that belong to this category
            if guess_category_from_relpath(p) != category:
                continue
            # normalize: if path in set-lists omits category, add it
            rel = p if p.startswith(category + "/") else f"{category}/{p}"
            rel = canon_path(rel)
            rec = by_path.get(rel)
            if rec is None:
                missing += 1
                continue
            seq = (rec.get("sequence_id") or rec.get("seq_id")
                   or rec.get("sequence_name") or guess_seq_from_relpath(rel))
            # keep only the keys your loader expects
            out[str(seq)].append({
                "filepath": rel,
                "R": rec.get("R") or rec.get("rotation") or rec.get("R_worldcam") or rec.get("R") or [[1,0,0],[0,1,0],[0,0,1]],
                "T": rec.get("T") or rec.get("translation") or rec.get("T_worldcam") or rec.get("T") or [0,0,0],
                "focal_length": rec.get("focal_length") or rec.get("focal") or rec.get("ndc_focal") or [1.0, 1.0],
                "principal_point": rec.get("principal_point") or rec.get("p0") or rec.get("ndc_p0") or [0.0, 0.0],
            })
        return out, missing

    train_dict, miss_tr = make_split(train_paths)
    test_dict,  miss_te = make_split(test_paths)

    if miss_tr or miss_te:
        print(f"[{category}] WARN: missing {miss_tr} train + {miss_te} test frames not found in frame_annotations.jgz")

    return train_dict, test_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CO3Dv2 root with category subfolders")
    ap.add_argument("--include_val_to", choices=["train", "test", "none"], default="test",
                    help="Fold 'val' frames into which split (default: test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.root
    cats = [d for d in os.listdir(root) if osp.isdir(osp.join(root, d))]

    total_train, total_test = 0, 0
    for cat in sorted(cats):
        cat_dir = osp.join(root, cat)
        fa_path = osp.join(cat_dir, "frame_annotations.jgz")
        sl_dir  = osp.join(cat_dir, "set_lists")
        if not (osp.isfile(fa_path) and osp.isdir(sl_dir)):
            continue

        print(f"[{cat}] reading frame_annotations.jgz …")
        # Build
        fold = None if args.include_val_to == "none" else args.include_val_to
        train_dict, test_dict = build_cat_splits(root, cat, include_val_to=fold)

        print(f"[{cat}] set_lists → train_seqs:{len(train_dict)} test_seqs:{len(test_dict)}")

        out_train = osp.join(root, f"{cat}_train.jgz")
        out_test  = osp.join(root, f"{cat}_test.jgz")

        if (not args.overwrite) and (osp.exists(out_train) or osp.exists(out_test)):
            print(f"[{cat}] outputs exist, use --overwrite to replace")
        else:
            write_json_gz(train_dict, out_train)
            write_json_gz(test_dict, out_test)
            kept_tr = sum(len(v) for v in train_dict.values())
            kept_te = sum(len(v) for v in test_dict.values())
            print(f"[{cat}] wrote {out_train} (frames:{kept_tr}, seqs:{len(train_dict)})")
            print(f"[{cat}] wrote {out_test}  (frames:{kept_te}, seqs:{len(test_dict)})")
            total_train += len(train_dict)
            total_test  += len(test_dict)

    print(f"Done. Total sequences written  train={total_train}  test={total_test}")

if __name__ == "__main__":
    main()

###
# python co3d_build_split_jgz.py --root /data/wanghaoxuan/CO3Dv2_single_seq --include_val_to test --overwrite
###