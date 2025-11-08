import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    label: str


def iou(a: Box, b: Box) -> float:
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    inter_w = max(0, min(ax1, bx1) - max(a.x, b.x))
    inter_h = max(0, min(ay1, by1) - max(a.y, b.y))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    ua = a.w * a.h + b.w * b.h - inter
    return inter / max(1, ua)


def greedy_match(preds: List[Box], gts: List[Box], iou_thr: float) -> Tuple[List[int], List[int]]:
    matches_pred = [-1] * len(preds)
    matches_gt = [-1] * len(gts)
    # sort pairs by IoU desc
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p.label != g.label:
                continue
            s = iou(p, g)
            if s >= iou_thr:
                pairs.append((s, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for _, i, j in pairs:
        if matches_pred[i] == -1 and matches_gt[j] == -1:
            matches_pred[i] = j
            matches_gt[j] = i
    return matches_pred, matches_gt


def evaluate_one(pred_json: str, gt_json: str, iou_thr: float) -> Dict[str, float]:
    with open(pred_json) as f:
        preds_raw = json.load(f)
    with open(gt_json) as f:
        gts_raw = json.load(f)
    preds = [Box(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), r.get("label", "text")) for r in preds_raw]
    gts = [Box(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]), r["label"]) for r in gts_raw]

    labels = sorted(set([r.label for r in preds] + [r.label for r in gts]))
    res: Dict[str, float] = {}
    for lbl in labels:
        p_lbl = [p for p in preds if p.label == lbl]
        g_lbl = [g for g in gts if g.label == lbl]
        mp, mg = greedy_match(p_lbl, g_lbl, iou_thr)
        tp = sum(1 for m in mp if m != -1)
        fp = sum(1 for m in mp if m == -1)
        fn = sum(1 for m in mg if m == -1)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        res[f"precision_{lbl}"] = prec
        res[f"recall_{lbl}"] = rec
        res[f"f1_{lbl}"] = f1
    # macro averages
    keys_p = [k for k in res.keys() if k.startswith("precision_")]
    keys_r = [k for k in res.keys() if k.startswith("recall_")]
    keys_f = [k for k in res.keys() if k.startswith("f1_")]
    res["precision_macro"] = float(np.mean([res[k] for k in keys_p])) if keys_p else 0.0
    res["recall_macro"] = float(np.mean([res[k] for k in keys_r])) if keys_r else 0.0
    res["f1_macro"] = float(np.mean([res[k] for k in keys_f])) if keys_f else 0.0
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="pred json file or dir")
    parser.add_argument("--gt", type=str, required=True, help="gt json file or dir")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    if os.path.isdir(args.pred):
        files = sorted([f for f in os.listdir(args.pred) if f.endswith('.json')])
        agg: Dict[str, List[float]] = {}
        for f in files:
            pr = os.path.join(args.pred, f)
            gr = os.path.join(args.gt, f)
            if not os.path.exists(gr):
                continue
            res = evaluate_one(pr, gr, args.iou)
            for k, v in res.items():
                agg.setdefault(k, []).append(v)
        summary = {k: float(np.mean(v)) for k, v in agg.items()}
        print(json.dumps(summary, indent=2))
    else:
        res = evaluate_one(args.pred, args.gt, args.iou)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


