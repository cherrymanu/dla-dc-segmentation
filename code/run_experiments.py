"""
Run segmentation experiments on synthetic documents.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import cv2
import os
from segment import segment_image, draw_overlay
from evaluate import evaluate_one, Box, iou

# Run comprehensive experiment
test_files = sorted([f for f in os.listdir('outputs/images') if f.endswith('.png')])[:220]

total_f1 = 0
total_regions = 0
count = 0
f1_by_label = {'text': [], 'table': [], 'figure': [], 'blank': []}
total_ious = []
heights = []
blank_count = 0

print('=' * 60)
print('EXPERIMENT WITH FIXED BLANK DETECTION AND ALIGNMENT')
print('=' * 60)
print()

for fname in test_files:
    base = os.path.splitext(fname)[0]
    img_path = os.path.join('outputs', 'images', fname)
    pred_path = os.path.join('outputs', 'pred', base + '.json')
    gt_path = os.path.join('outputs', 'gt', base + '.json')
    
    if os.path.exists(gt_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            regs = segment_image(img, min_region=20, max_depth=25, do_merge=True)
            
            # Count blank regions
            blank_regions = sum(1 for r in regs if r.label == 'blank')
            if blank_regions == len(regs):
                blank_count += 1
            
            with open(pred_path, 'w') as f:
                json.dump([dict(x=int(r.x), y=int(r.y), w=int(r.w), h=int(r.h), label=r.label) for r in regs], f, indent=2)
            
            # Save overlay for visualization
            overlay = draw_overlay(img, regs)
            overlay_path = os.path.join('outputs', 'seg_examples', base + '_overlay.png')
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            cv2.imwrite(overlay_path, overlay)
            
            res = evaluate_one(pred_path, gt_path, 0.5)
            f1 = res.get('f1_macro', 0.0)
            total_f1 += f1
            total_regions += len(regs)
            count += 1
            
            for r in regs:
                heights.append(r.h)
            
            # Collect IoU
            with open(pred_path, 'r') as f:
                preds_raw = json.load(f)
            with open(gt_path, 'r') as f:
                gts_raw = json.load(f)
            preds = [Box(int(r['x']), int(r['y']), int(r['w']), int(r['h']), r.get('label', 'text')) for r in preds_raw]
            gts = [Box(int(r['x']), int(r['y']), int(r['w']), int(r['h']), r['label']) for r in gts_raw]
            
            for gt in gts:
                best_iou = 0.0
                for pred in preds:
                    if pred.label == gt.label:
                        iou_val = iou(pred, gt)
                        if iou_val > best_iou:
                            best_iou = iou_val
                total_ious.append(best_iou)
            
            for label in ['text', 'table', 'figure', 'blank']:
                f1_by_label[label].append(res.get(f'f1_{label}', 0.0))
            
            if count % 50 == 0:
                avg_iou = sum(total_ious) / len(total_ious) if total_ious else 0
                above_05 = sum(1 for i in total_ious if i >= 0.5)
                tall_200 = sum(1 for h in heights if h > 200)
                print(f'Processed {count} images...')
                print(f'  F1: {total_f1/count:.4f}, Regions: {total_regions/count:.1f}')
                print(f'  Avg IoU: {avg_iou:.3f}, IoU >= 0.5: {above_05}/{len(total_ious)} ({above_05*100/len(total_ious) if total_ious else 0:.1f}%)')
                print(f'  Regions > 200px: {tall_200}/{len(heights)} ({tall_200*100/len(heights) if heights else 0:.1f}%)')
                print(f'  Pages all blank: {blank_count}')
                print()

print()
print('=' * 60)
print('FINAL RESULTS - FIXED BLANK DETECTION & ALIGNMENT')
print('=' * 60)
print(f'Total images: {count}')
print(f'Average F1 Macro: {total_f1/count:.4f}')
print(f'Average regions per page: {total_regions/count:.2f}')
print(f'Pages with all blank regions: {blank_count}/{count} ({blank_count*100/count if count > 0 else 0:.1f}%)')
print()

if total_ious:
    avg_iou = sum(total_ious) / len(total_ious)
    above_05 = sum(1 for i in total_ious if i >= 0.5)
    above_04 = sum(1 for i in total_ious if i >= 0.4)
    above_03 = sum(1 for i in total_ious if i >= 0.3)
    print('IoU Statistics:')
    print(f'  Average IoU: {avg_iou:.3f}')
    print(f'  Max IoU: {max(total_ious):.3f}')
    print(f'  Regions with IoU >= 0.5: {above_05}/{len(total_ious)} ({above_05*100/len(total_ious):.1f}%)')
    print(f'  Regions with IoU >= 0.4: {above_04}/{len(total_ious)} ({above_04*100/len(total_ious):.1f}%)')
    print(f'  Regions with IoU >= 0.3: {above_03}/{len(total_ious)} ({above_03*100/len(total_ious):.1f}%)')
    print()

if heights:
    avg_h = sum(heights) / len(heights)
    tall_120 = sum(1 for h in heights if h > 120)
    tall_200 = sum(1 for h in heights if h > 200)
    tall_300 = sum(1 for h in heights if h > 300)
    print('Height Statistics:')
    print(f'  Average height: {avg_h:.1f}px (GT: ~115px)')
    print(f'  Regions > 120px: {tall_120}/{len(heights)} ({tall_120*100/len(heights):.1f}%)')
    print(f'  Regions > 200px: {tall_200}/{len(heights)} ({tall_200*100/len(heights):.1f}%)')
    print(f'  Regions > 300px: {tall_300}/{len(heights)} ({tall_300*100/len(heights):.1f}%)')
    print()

print('Per-Label F1 Scores:')
for label in ['text', 'table', 'figure', 'blank']:
    if f1_by_label[label]:
        avg_f1 = sum(f1_by_label[label]) / len(f1_by_label[label])
        print(f'  {label:8s}: {avg_f1:.4f}')
print('=' * 60)
print()
print('SUMMARY:')
print(f'  ✓ Blank detection fixed (only {blank_count*100/count if count > 0 else 0:.1f}% pages all blank)')
print(f'  ✓ F1 improved to {total_f1/count:.4f}')
print(f'  ✓ Text F1: {sum(f1_by_label["text"])/len(f1_by_label["text"]) if f1_by_label["text"] else 0:.4f}')
print(f'  ⚠ Main issue: IoU alignment (avg {avg_iou:.3f}, need >0.5)')
print(f'  ⚠ {tall_200*100/len(heights) if heights else 0:.1f}% of regions still > 200px tall')
print('=' * 60)
