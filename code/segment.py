import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import cv2


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    label: Optional[str] = None

    def to_bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _projection_valleys(binary: np.ndarray, axis: int) -> List[int]:
    """
    Improved whitespace detection for better content boundary identification.
    Uses adaptive thresholding and gap analysis to find significant whitespace regions.
    """
    # axis=0 -> vertical projection (sum over rows) => columns
    # axis=1 -> horizontal projection (sum over cols) => rows
    # Projection: (255 - binary) gives HIGH values for whitespace, LOW for content
    proj = (255 - binary).sum(axis=axis).astype(np.float32)
    
    # Normalize to [0, 1]
    proj_min, proj_max = proj.min(), proj.max()
    if proj_max - proj_min < 1e-6:
        return []  # No variation
    
    proj_norm = (proj - proj_min) / (proj_max - proj_min)
    
    # Improved adaptive thresholding: better detect significant gaps
    # Use percentile-based thresholds that adapt to content density
    # Higher percentiles for sparse content, lower for dense content
    content_density = 1.0 - np.mean(proj_norm)  # How much content vs whitespace
    
    # For very sparse content, use fixed thresholds to avoid percentile issues
    # When most values are 1.0, percentiles don't help distinguish gaps
    if content_density < 0.2:  # Very sparse content (lots of whitespace)
        # Use fixed thresholds that are high enough to catch gaps but not too high
        # Look for values that are significantly above the mean (gaps are whiter)
        mean_val = np.mean(proj_norm)
        std_val = np.std(proj_norm)
        thresh_large = min(0.95, mean_val + 0.5 * std_val)  # At least 0.5 std above mean
        thresh_small = min(0.90, mean_val + 0.3 * std_val)  # At least 0.3 std above mean
        # Ensure thresholds are reasonable
        thresh_large = max(0.85, thresh_large)
        thresh_small = max(0.80, thresh_small)
    elif content_density < 0.3:  # Sparse content (lots of whitespace)
        thresh_large = max(0.85, np.percentile(proj_norm, 80))
        thresh_small = max(0.70, np.percentile(proj_norm, 65))
    elif content_density < 0.6:  # Moderate content
        thresh_large = max(0.80, np.percentile(proj_norm, 75))
        thresh_small = max(0.65, np.percentile(proj_norm, 60))
    else:  # Dense content (little whitespace)
        thresh_large = max(0.75, np.percentile(proj_norm, 70))
        thresh_small = max(0.60, np.percentile(proj_norm, 55))
    
    # Adaptive gap sizes based on image dimension
    length = len(proj_norm)
    min_gap_size_large = max(5, length // 30)  # Large gaps: ~3% of dimension
    min_gap_size_small = max(3, length // 50)  # Small gaps: ~2% of dimension
    
    peaks = []
    i = 0
    while i < len(proj_norm):
        if proj_norm[i] > thresh_large:
            # Large gap: significant whitespace between major content blocks
            start = i
            while i < len(proj_norm) and proj_norm[i] > thresh_large:
                i += 1
            end = i
            gap_size = end - start
            
            # Check gap quality: should be consistently high
            gap_mean = np.mean(proj_norm[start:end]) if end > start else 0
            if gap_size >= min_gap_size_large and gap_mean > thresh_large * 0.9:
                center = (start + end) // 2
                peaks.append(center)
        elif proj_norm[i] > thresh_small:
            # Smaller but consistent gap (e.g., table boundaries, paragraph breaks)
            start = i
            consistent_high = 0
            while i < len(proj_norm) and proj_norm[i] > thresh_small:
                if proj_norm[i] > thresh_small * 1.1:  # Significantly above threshold
                    consistent_high += 1
                i += 1
            end = i
            gap_size = end - start
            
            # Keep if gap is large enough and has consistent high values
            # This helps catch table boundaries and content separations
            if gap_size >= min_gap_size_small and consistent_high >= gap_size * 0.5:
                center = (start + end) // 2
                # Avoid duplicates: only add if not too close to existing peaks
                min_distance = max(5, length // 20)  # At least 5% of dimension apart
                if not peaks or abs(center - peaks[-1]) > min_distance:
                    peaks.append(center)
        else:
            i += 1
    
    return peaks


def _split_indices_from_valleys(valleys: List[int], length: int, min_size: int) -> List[Tuple[int, int]]:
    if not valleys:
        return [(0, length)]
    # Group valleys that are close together (within gap_threshold pixels)
    # This prevents splitting at every line break in text
    # Use a more aggressive threshold to find more split points
    # Lower threshold = more splits = better segmentation of content blocks
    gap_threshold = max(min_size // 6, 10)  # Even more aggressive: smaller threshold = more splits
    splits = []
    start = 0
    i = 0
    while i < len(valleys):
        # Find the end of the current valley group
        # Valleys are grouped if they're within gap_threshold pixels of each other
        j = i
        valley_group_start = valleys[i]
        valley_group_end = valleys[i]
        while j + 1 < len(valleys):
            # Check if next valley is close enough to be in the same group
            if valleys[j + 1] - valley_group_end <= gap_threshold:
                valley_group_end = valleys[j + 1]
                j += 1
            else:
                break
        # Cut at the middle of the valley group
        cut = (valley_group_start + valley_group_end) // 2
        # Use smaller min_size to allow more splits
        min_split_size = max(min_size // 2, 20)  # Allow smaller splits
        if cut - start >= min_split_size:
            splits.append((start, cut))
            start = cut
        i = j + 1
    # Use smaller min_size for final split
    min_split_size = max(min_size // 2, 20)
    if length - start >= min_split_size:
        splits.append((start, length))
    if not splits:
        return [(0, length)]
    return splits


def _compute_region_features(binary: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute feature vector for a region: (intensity_variance, texture_uniformity, edge_density)
    Returns: (variance, texture_uniformity, edge_density)
    """
    h, w = binary.shape
    if h == 0 or w == 0:
        return (0.0, 0.0, 0.0)
    
    # 1. Intensity variance (variance of pixel intensities)
    intensity_variance = float(binary.var())
    
    # 2. Texture uniformity (via projection variance - lower variance = more uniform)
    # Compute horizontal and vertical projection variances
    proj_h = (255 - binary).sum(axis=1).astype(np.float32)
    proj_v = (255 - binary).sum(axis=0).astype(np.float32)
    proj_h_var = float(proj_h.var()) if len(proj_h) > 1 else 0.0
    proj_v_var = float(proj_v.var()) if len(proj_v) > 1 else 0.0
    texture_uniformity = 1.0 / (1.0 + (proj_h_var + proj_v_var) / (h + w))  # Higher = more uniform
    
    # 3. Edge density
    edges = cv2.Canny(binary, 50, 150)
    edge_density = float(edges.mean())
    
    return (intensity_variance, texture_uniformity, edge_density)


def _feature_homogeneity_score(features1: Tuple[float, float, float], 
                                features2: Tuple[float, float, float]) -> float:
    """
    Compute how different two feature vectors are (higher = more different = better split point).
    Uses Euclidean distance in normalized feature space.
    """
    # Normalize features to [0, 1] range (rough normalization)
    # Intensity variance: typically 0-65025 (255^2), normalize by max
    # Texture uniformity: already 0-1
    # Edge density: 0-1 (already normalized)
    f1_norm = (
        min(features1[0] / 65025.0, 1.0),  # intensity variance
        features1[1],  # texture uniformity
        features1[2]   # edge density
    )
    f2_norm = (
        min(features2[0] / 65025.0, 1.0),
        features2[1],
        features2[2]
    )
    
    # Euclidean distance (higher = more different = better split)
    diff = sum((a - b) ** 2 for a, b in zip(f1_norm, f2_norm)) ** 0.5
    return diff


def _find_best_feature_split(binary: np.ndarray, axis: int, min_wh: int) -> Optional[int]:
    """
    Find the best split point along given axis using feature homogeneity.
    axis=0: vertical split (split columns)
    axis=1: horizontal split (split rows)
    Returns: best split index, or None if no good split found.
    """
    h, w = binary.shape
    if axis == 0:  # vertical split
        length = w
        min_split = min_wh
        max_split = w - min_wh
    else:  # horizontal split
        length = h
        min_split = min_wh
        max_split = h - min_wh
    
    if max_split <= min_split:
        return None
    
    # Try candidate splits (sample every few pixels to avoid being too slow)
    # Use more candidate splits for better accuracy
    step = max(1, length // 30)  # Try ~30 candidate splits for better precision
    best_score = -1.0
    best_idx = None
    
    for split_idx in range(min_split, max_split, step):
        if axis == 0:  # vertical split at column split_idx
            left = binary[:, :split_idx]
            right = binary[:, split_idx:]
        else:  # horizontal split at row split_idx
            left = binary[:split_idx, :]
            right = binary[split_idx:, :]
        
        if left.size == 0 or right.size == 0:
            continue
        
        # Compute features for both halves
        features_left = _compute_region_features(left)
        features_right = _compute_region_features(right)
        
        # Score: how different are the halves? (higher = better split)
        score = _feature_homogeneity_score(features_left, features_right)
        
        if score > best_score:
            best_score = score
            best_idx = split_idx
    
    # Only return if the difference is significant enough
    # Lower threshold to allow more splits - even small differences can be meaningful
    if best_score > 0.05:  # Threshold: halves must be meaningfully different
        return best_idx
    return None


def _feature_stop(binary: np.ndarray, min_wh: int) -> bool:
    """
    Determine if splitting should stop for this region.
    Only stop if region is truly uniform (very small, or mostly blank).
    Be more aggressive about continuing to split large regions.
    """
    h, w = binary.shape
    # Always stop if region is too small
    if w < min_wh or h < min_wh:
        return True
    
    # For larger regions, be more aggressive about continuing to split
    # Only stop if region is very small OR mostly blank
    # Don't stop just because edge density is low - large text blocks can have low edge density
    
    # Check if region is mostly blank (very high white ratio)
    white_pixels = np.sum(binary < 50)  # Very dark pixels (content)
    total_pixels = binary.size
    content_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Only stop if region is very small OR almost completely blank
    if content_ratio < 0.02:  # Less than 2% content = mostly blank
        return True
    
    # For larger regions, continue splitting even if edge density is low
    # This allows splitting large text blocks into smaller paragraphs
    # Only stop if region is very small
    if w < min_wh * 2 and h < min_wh * 2:
        # Small region - check edge density
        edges = cv2.Canny(binary, 50, 150)
        density = edges.mean()
        if density < 0.01:  # mostly blank
            return True
    
    # For larger regions, always continue splitting
    return False


def _xy_cut(binary: np.ndarray, x: int, y: int, w: int, h: int, min_wh: int, depth: int, max_depth: int, use_feature_split: bool = True) -> List[Region]:
    roi = binary[y:y + h, x:x + w]
    if depth >= max_depth or _feature_stop(roi, min_wh=min_wh):
        return [Region(x, y, w, h)]

    if use_feature_split:
        # HYBRID APPROACH: Combine feature-homogeneity and whitespace-based splitting
        # This gives us the best of both worlds: content-aware splitting + whitespace detection
        
        # 1. Try feature-homogeneity based splitting
        v_split_feat = _find_best_feature_split(roi, axis=0, min_wh=min_wh)
        h_split_feat = _find_best_feature_split(roi, axis=1, min_wh=min_wh)
        
        # 2. Try whitespace-based splitting
        v_valleys = _projection_valleys(roi, axis=0)
        h_valleys = _projection_valleys(roi, axis=1)
        v_splits_ws = _split_indices_from_valleys(v_valleys, w, min_wh)
        h_splits_ws = _split_indices_from_valleys(h_valleys, h, min_wh)
        
        # 3. Get best whitespace split (middle of first significant gap)
        v_split_ws = None
        h_split_ws = None
        if len(v_splits_ws) > 1:
            # Use the first split point
            v_split_ws = v_splits_ws[0][1] if v_splits_ws[0][1] < w else None
        if len(h_splits_ws) > 1:
            h_split_ws = h_splits_ws[0][1] if h_splits_ws[0][1] < h else None
        
        # 4. Compute scores for feature-based splits
        v_score_feat = 0.0
        h_score_feat = 0.0
        if v_split_feat is not None:
            left = roi[:, :v_split_feat]
            right = roi[:, v_split_feat:]
            if left.size > 0 and right.size > 0:
                f_left = _compute_region_features(left)
                f_right = _compute_region_features(right)
                v_score_feat = _feature_homogeneity_score(f_left, f_right)
        
        if h_split_feat is not None:
            top = roi[:h_split_feat, :]
            bottom = roi[h_split_feat:, :]
            if top.size > 0 and bottom.size > 0:
                f_top = _compute_region_features(top)
                f_bottom = _compute_region_features(bottom)
                h_score_feat = _feature_homogeneity_score(f_top, f_bottom)
        
        # 5. Compute whitespace scores (how clear is the whitespace gap)
        v_score_ws = 0.0
        h_score_ws = 0.0
        if v_split_ws is not None:
            # Check projection value at split point (higher = more whitespace)
            proj_v = (255 - roi).sum(axis=0).astype(np.float32)
            if v_split_ws < len(proj_v):
                proj_norm = (proj_v - proj_v.min()) / (proj_v.max() - proj_v.min() + 1e-6)
                v_score_ws = float(proj_norm[v_split_ws])  # Higher = better whitespace gap
        
        if h_split_ws is not None:
            proj_h = (255 - roi).sum(axis=1).astype(np.float32)
            if h_split_ws < len(proj_h):
                proj_norm = (proj_h - proj_h.min()) / (proj_h.max() - proj_h.min() + 1e-6)
                h_score_ws = float(proj_norm[h_split_ws])
        
        # 6. Combine scores: weighted average of feature and whitespace signals
        # Feature score is more reliable when high, whitespace is more reliable when clear
        alpha = 0.6  # Weight for feature-homogeneity
        beta = 0.4   # Weight for whitespace
        
        v_score_combined = alpha * v_score_feat + beta * v_score_ws if (v_split_feat is not None or v_split_ws is not None) else 0.0
        h_score_combined = alpha * h_score_feat + beta * h_score_ws if (h_split_feat is not None or h_split_ws is not None) else 0.0
        
        # 7. Choose best split: use the one with higher combined score
        regions: List[Region] = []
        
        # Decide on vertical split: use whichever has better combined score
        v_split = None
        if v_split_feat is not None and v_split_ws is not None:
            # Both available - use the one with better combined score
            if v_score_combined > 0.1:  # Only if combined score is good
                # Prefer feature-based if score is high, otherwise whitespace
                v_split = v_split_feat if v_score_feat > 0.08 else v_split_ws
        elif v_split_feat is not None and v_score_feat > 0.05:
            v_split = v_split_feat
        elif v_split_ws is not None and v_score_ws > 0.3:  # Whitespace must be clear
            v_split = v_split_ws
        
        # Decide on horizontal split
        h_split = None
        if h_split_feat is not None and h_split_ws is not None:
            if h_score_combined > 0.1:
                h_split = h_split_feat if h_score_feat > 0.08 else h_split_ws
        elif h_split_feat is not None and h_score_feat > 0.05:
            h_split = h_split_feat
        elif h_split_ws is not None and h_score_ws > 0.3:
            h_split = h_split_ws
        
        # 8. Choose axis based on combined scores and balance
        if v_split is not None and h_split is not None:
            # Both available - choose based on combined score and balance
            v_balance = min(v_split, w - v_split) / max(v_split, w - v_split) if v_split > 0 else 0
            h_balance = min(h_split, h - h_split) / max(h_split, h - h_split) if h_split > 0 else 0
            v_weighted = v_score_combined * (0.7 + 0.3 * v_balance)
            h_weighted = h_score_combined * (0.7 + 0.3 * h_balance)
            
            if v_weighted > h_weighted:
                # Vertical split
                regions.extend(_xy_cut(binary, x, y, v_split, h, min_wh, depth + 1, max_depth, use_feature_split))
                regions.extend(_xy_cut(binary, x + v_split, y, w - v_split, h, min_wh, depth + 1, max_depth, use_feature_split))
                return regions
            else:
                # Horizontal split
                regions.extend(_xy_cut(binary, x, y, w, h_split, min_wh, depth + 1, max_depth, use_feature_split))
                regions.extend(_xy_cut(binary, x, y + h_split, w, h - h_split, min_wh, depth + 1, max_depth, use_feature_split))
                return regions
        elif v_split is not None:
            # Only vertical split
            regions.extend(_xy_cut(binary, x, y, v_split, h, min_wh, depth + 1, max_depth, use_feature_split))
            regions.extend(_xy_cut(binary, x + v_split, y, w - v_split, h, min_wh, depth + 1, max_depth, use_feature_split))
            return regions
        elif h_split is not None:
            # Only horizontal split
            regions.extend(_xy_cut(binary, x, y, w, h_split, min_wh, depth + 1, max_depth, use_feature_split))
            regions.extend(_xy_cut(binary, x, y + h_split, w, h - h_split, min_wh, depth + 1, max_depth, use_feature_split))
            return regions
        
        # No good split found
        return [Region(x, y, w, h)]
    else:
        # WHITESPACE-BASED SPLITTING with content-aware improvements
        # Compute projection valleys on both axes
        v_valleys = _projection_valleys(roi, axis=0)
        h_valleys = _projection_valleys(roi, axis=1)

        # Prefer the split that yields more balanced pieces
        v_splits = _split_indices_from_valleys(v_valleys, w, min_wh)
        h_splits = _split_indices_from_valleys(h_valleys, h, min_wh)

        # CONTENT-AWARE SPLITTING: Detect table structures and avoid splitting through them
        # This prevents over-segmentation of tables and improves bounding box accuracy
        
        # Detect potential table structure: check for grid-like patterns
        proj_h = (255 - roi).sum(axis=1).astype(np.float32)
        proj_v = (255 - roi).sum(axis=0).astype(np.float32)
        
        # Improved periodicity detection for tables
        # Make it stricter to avoid false positives from text lines
        def has_periodicity(proj, min_periods=3):
            if len(proj) < 8:
                return False
            proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
            # Count peaks (high values) - tables have regular peaks from grid lines
            # Use higher threshold to avoid text line false positives
            peaks = np.sum(proj_norm > 0.6)  # Higher threshold
            # Also check for regularity: variance of peak spacing
            peak_indices = np.where(proj_norm > 0.6)[0]
            if len(peak_indices) >= min_periods:
                if len(peak_indices) > 1:
                    spacing = np.diff(peak_indices)
                    spacing_cv = np.std(spacing) / (np.mean(spacing) + 1e-6)  # Coefficient of variation
                    # Regular spacing (low CV) indicates table structure
                    # Make CV threshold stricter
                    return peaks >= min_periods and spacing_cv < 0.4
                return peaks >= min_periods
            return False
        
        # Only detect table structure if BOTH directions have strong periodicity
        # This prevents text (which has periodicity in one direction) from being detected as table
        has_table_structure = has_periodicity(proj_h, 4) and has_periodicity(proj_v, 3)  # Stricter requirements
        
        # CONTENT-AWARE SPLITTING DECISION
        # Temporarily disable table-preserving logic to ensure splitting works
        # TODO: Re-enable with better conditions after verifying splitting works
        # Only apply table-preserving logic if:
        # 1. We're not at the top level (depth > 0) - always split at top level
        # 2. Region is small enough to be a single table (not the entire page)
        # 3. AND has clear table structure
        # This prevents the entire page from being treated as one table
        # At top level (depth=0), always allow splitting to break up the page
        is_small_table_region = w < 500 and h < 700  # Reasonable table size limit
        is_not_top_level = depth > 0  # Don't prevent splitting at top level
        
        # Temporarily disabled to debug splitting
        if False and has_table_structure and is_small_table_region and is_not_top_level:
            # For small table regions: be conservative about splitting
            # Only split if there's a VERY clear gap (large whitespace)
            # This prevents breaking up table structures
            
            # Filter splits to only keep very large gaps
            v_splits_filtered = []
            h_splits_filtered = []
            
            # Check if splits are at significant whitespace boundaries
            for split_start, split_end in v_splits:
                if split_end < w:
                    # Check gap size at split point
                    gap_size = split_end - split_start
                    # For tables, only split at very large gaps (>15% of width)
                    if gap_size > w * 0.15:
                        v_splits_filtered.append((split_start, split_end))
            
            for split_start, split_end in h_splits:
                if split_end < h:
                    gap_size = split_end - split_start
                    # For tables, only split at very large gaps (>15% of height)
                    if gap_size > h * 0.15:
                        h_splits_filtered.append((split_start, split_end))
            
            # Use filtered splits if they exist, otherwise proceed with normal splitting
            # Don't prevent all splits - just be more conservative
            if len(v_splits_filtered) > 1:
                v_splits = v_splits_filtered
            if len(h_splits_filtered) > 1:
                h_splits = h_splits_filtered
        
        # For all regions, use improved splitting heuristic
        # Choose split direction based on content structure
        if len(h_splits) > len(v_splits):
            use_vertical = False
        elif len(v_splits) > len(h_splits):
            use_vertical = True
        else:
            # If equal number of splits, prefer the one that creates more balanced aspect ratios
            # For tall images (h > w), prefer horizontal splits
            use_vertical = len(v_splits) * h <= len(h_splits) * w

    regions: List[Region] = []
    if use_vertical and len(v_splits) > 1:
        for xs, xe in v_splits:
                regions.extend(_xy_cut(binary, x + xs, y, xe - xs, h, min_wh, depth + 1, max_depth, use_feature_split))
        return regions
    if (not use_vertical) and len(h_splits) > 1:
        for ys, ye in h_splits:
                regions.extend(_xy_cut(binary, x, y + ys, w, ye - ys, min_wh, depth + 1, max_depth, use_feature_split))
        return regions

        # If no splits found but region is still large, try to find smaller gaps
        # This helps split large columns into smaller content blocks
        if w > min_wh * 3 or h > min_wh * 3:  # Region is still quite large
            # Try to find smaller gaps with lower thresholds
            v_valleys_fine = _projection_valleys(roi, axis=0)
            h_valleys_fine = _projection_valleys(roi, axis=1)
            
            # If we found valleys but they didn't create splits, try with smaller min_size
            if len(v_valleys_fine) > 0 and len(v_splits) <= 1:
                v_splits_fine = _split_indices_from_valleys(v_valleys_fine, w, min_wh // 2)  # Smaller min_size
                if len(v_splits_fine) > 1:
                    for xs, xe in v_splits_fine:
                        regions.extend(_xy_cut(binary, x + xs, y, xe - xs, h, min_wh, depth + 1, max_depth, use_feature_split))
                    return regions  # Only return if we actually found splits
            
            if len(h_valleys_fine) > 0 and len(h_splits) <= 1:
                h_splits_fine = _split_indices_from_valleys(h_valleys_fine, h, min_wh // 2)  # Smaller min_size
                if len(h_splits_fine) > 1:
                    for ys, ye in h_splits_fine:
                        regions.extend(_xy_cut(binary, x, y + ys, w, ye - ys, min_wh, depth + 1, max_depth, use_feature_split))
                    return regions  # Only return if we actually found splits

        # If we get here, no splits were found - return the region as-is
    return [Region(x, y, w, h)]


def _label_region(gray: np.ndarray, region: Region) -> str:
    x, y, w, h = region.to_bbox()
    roi = gray[y:y + h, x:x + w]
    if w == 0 or h == 0:
        return "blank"
    
    # Detect and crop margins (white borders) to focus on content
    # Find the bounding box of non-white content
    # White pixels are typically > 250 in grayscale
    white_thresh = 250
    non_white = roi < white_thresh
    
    if np.sum(non_white) == 0:
        return "blank"  # Entire region is white
    
    # Find bounding box of content (non-white area)
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return "blank"
    
    y_start, y_end = np.where(rows)[0][[0, -1]]
    x_start, x_end = np.where(cols)[0][[0, -1]]
    
    # Crop to content area, being more aggressive about excluding white borders
    # Use the detected content bounding box directly, with minimal padding
    # This helps when regions include gaps or large white areas
    padding = 5  # Small padding to avoid cutting into content
    min_x = max(0, x_start - padding)
    min_y = max(0, y_start - padding)
    max_x = min(w, x_end + 1 + padding)
    max_y = min(h, y_end + 1 + padding)
    
    roi_cropped = roi[min_y:max_y, min_x:max_x]
    
    # If cropped region is too small (< 20% of original), use a more conservative crop
    # This handles edge cases where content detection might be too aggressive
    if roi_cropped.size < roi.size * 0.2:
        # Use a more conservative crop: exclude outer 15% on each side
        margin_pct = 0.15
        min_x = int(w * margin_pct)
        min_y = int(h * margin_pct)
        max_x = int(w * (1 - margin_pct))
        max_y = int(h * (1 - margin_pct))
        roi_cropped = roi[min_y:max_y, min_x:max_x]
    
    # Resize for analysis
    cw, ch = roi_cropped.shape[1], roi_cropped.shape[0]
    roi_small = cv2.resize(roi_cropped, (max(1, cw // 4), max(1, ch // 4)))
    edges = cv2.Canny(roi_small, 50, 150)
    edge_density = edges.mean()

    # Horizontal/vertical stroke periodicity via projection FFT peakiness
    proj_h = (255 - roi_small).sum(axis=1).astype(np.float32)
    proj_v = (255 - roi_small).sum(axis=0).astype(np.float32)
    def peakiness(proj):
        if proj.size < 8:
            return 0.0
        f = np.fft.rfft((proj - proj.mean()) / (proj.std() + 1e-6))
        mag = np.abs(f)
        # skip DC and highest bins
        core = mag[2:max(3, len(mag)//2)]
        if core.size == 0:
            return 0.0
        return float(core.max() / (core.mean() + 1e-6))

    ph = peakiness(proj_h)
    pv = peakiness(proj_v)

    # Improved blank detection: Check multiple criteria
    # Blank regions have very low edge density and are mostly white
    # Key: Blank regions should be checked FIRST before other labels
    white_ratio = np.sum(roi_cropped > 240) / roi_cropped.size if roi_cropped.size > 0 else 0
    very_white_ratio = np.sum(roi_cropped > 250) / roi_cropped.size if roi_cropped.size > 0 else 0
    intensity_variance = float(roi_cropped.var())
    
    # Check for blank: very low edge density AND mostly white
    # Use multiple thresholds to catch different types of blank regions
    is_blank = False
    
    # Primary check: very low edge density (blank regions have almost no edges)
    # edge_density from edges.mean() is 0-255 range, normalize to 0-1
    edge_density_norm = edge_density / 255.0
    
    # Very blank: extremely low edge density (< 2% of pixels are edges)
    if edge_density_norm < 0.02:
        is_blank = True
    # Mostly blank: low edge density and high white ratio
    elif edge_density_norm < 0.08 and white_ratio > 0.70:  # Low edges + mostly white (more lenient)
        is_blank = True
    # Very white: extremely high white ratio even with some edges (more lenient)
    elif very_white_ratio > 0.75 and edge_density_norm < 0.15:  # Very white + moderate edges
        is_blank = True
    # Low variance + mostly white: blank regions have uniform intensity
    elif intensity_variance < 500 and white_ratio > 0.65:  # Low variance + mostly white (more lenient)
        is_blank = True
    # Additional check: if region is very large and mostly white, likely blank
    if roi_cropped.size > 5000 and white_ratio > 0.70 and edge_density_norm < 0.10:
        is_blank = True
    # Special case: if region is mostly white (>75%) and not too many edges, likely blank
    # (even if variance is higher, if it's very white, it's probably blank)
    if white_ratio > 0.75 and edge_density_norm < 0.20:  # Mostly white + not too many edges
        is_blank = True
    
    if is_blank:
        return "blank"
    
    # Table detection: Detect grid structures
    # Tables have grid lines in BOTH directions (horizontal and vertical)
    # Use edge-based detection as primary method (more reliable than periodicity for small tables)
    is_table = False
    
    # Check edge structure: tables have edges in both directions (grid lines)
    edges_h = np.sum(edges, axis=1)
    edges_v = np.sum(edges, axis=0)
    
    # Find strong horizontal and vertical edge lines (grid lines)
    h_edge_thresh = edges_h.mean() * 1.5  # Threshold for horizontal grid lines
    v_edge_thresh = edges_v.mean() * 1.5  # Threshold for vertical grid lines
    
    h_lines = np.sum(edges_h > h_edge_thresh)  # Number of strong horizontal lines
    v_lines = np.sum(edges_v > v_edge_thresh)  # Number of strong vertical lines
    
    # Tables should have multiple grid lines in both directions
    # Lower threshold: at least 2 lines in each direction (for small tables)
    if h_lines >= 2 and v_lines >= 2:
        # Check edge balance: both directions should have similar edge strength
        edge_balance = min(edges_h.mean(), edges_v.mean()) / max(edges_h.mean(), edges_v.mean()) if max(edges_h.mean(), edges_v.mean()) > 0 else 0
        if edge_balance > 0.2:  # Some balance (grid structure)
            is_table = True
    
    # Also check periodicity method for larger tables
    if not is_table and ph > 1.5 and pv > 1.5:  # Lower threshold
        periodicity_balance = min(ph, pv) / max(ph, pv) if max(ph, pv) > 0 else 0
        edge_balance = min(edges_h.mean(), edges_v.mean()) / max(edges_h.mean(), edges_v.mean()) if max(edges_h.mean(), edges_v.mean()) > 0 else 0
        
        # Table: periodicity in both directions with reasonable balance
        if periodicity_balance > 0.3 and edge_balance > 0.2:
            is_table = True
    
    if is_table:
        return "table"
    
    # Text detection: DEFAULT - most content is text
    # Text has horizontal periodicity (lines) but NOT vertical periodicity (no grid)
    # Be very lenient - if it's not blank, table, or figure, it's text
    
    # Strong text: high horizontal periodicity, low vertical (no grid)
    if ph > 1.5 and pv < 1.8:  # Horizontal lines, no vertical grid
        return "text"
    
    # Moderate text: some horizontal periodicity
    if ph > 1.2 and pv < 2.0:
        return "text"
    
    # Weak text: has horizontal structure but not table-like
    if ph > 1.0 and pv < 2.5:  # Some horizontal structure, not a grid
        return "text"
    
    # Default: if it's not clearly a table (grid), it's text
    # Tables require BOTH ph > 2.0 AND pv > 2.0, so if that's not met, it's text
    if not (ph > 2.0 and pv > 2.0):  # Not a clear grid = text
        return "text"
    
    # Even if periodicity is high, if it's not balanced (grid), it's text
    if ph > 2.0 or pv > 2.0:  # High in one direction = text lines, not grid
        return "text"
    
    # Figure detection: ONLY for extremely rare, clear cases
    # Figures should be < 10% of regions - they're rare in documents
    # Must have ALL of these characteristics simultaneously:
    # 1. Very high edge density (> 0.35) - lots of edges/features
    # 2. Very low periodicity in BOTH directions (< 0.9) - completely chaotic, no structure
    # 3. Very high variance (> 800) - highly varied content
    # 4. NOT text-like (ph < 0.9) - no horizontal lines
    # 5. NOT table-like (no grid structure - already checked above, is_table is False)
    
    # Only label as figure if it's ABSOLUTELY clear:
    is_figure = False
    
    # Primary check: high edge density with low periodicity (chaotic content)
    # edge_density from edges.mean() is 0-255 range, normalize to 0-1
    edge_density_norm_check = edge_density / 255.0
    
    # Figures have high edge density (lots of features) but low periodicity (no regular structure)
    # Lower thresholds to detect real figures
    if (edge_density_norm_check > 0.15 and     # High edge density (lowered from 0.35)
        ph < 1.2 and pv < 1.2 and              # Low periodicity (chaotic, not text/table)
        intensity_variance > 400 and            # High variance (varied content, lowered from 800)
        not is_table):                         # Not table (already checked)
        is_figure = True
    
    # Strong figure: very high edge density with very low periodicity
    if (edge_density_norm_check > 0.25 and     # Very high edge density (lowered from 0.45)
        ph < 1.0 and pv < 1.0 and              # Very low periodicity (lowered from 0.7)
        intensity_variance > 600 and           # Very high variance (lowered from 1000)
        not is_table):                         # Not table
        is_figure = True
    
    if is_figure:
        return "figure"
    
    # Final fallback: default to text for everything else
    # This prevents over-classification as figures
    # Most regions in documents are text, so this is the safe default
    return "text"


def _trim_margins(gray: np.ndarray, region: Region, white_thresh: int = 250) -> Region:
    """Trim white margins from a region to focus on actual content."""
    x, y, w, h = region.to_bbox()
    roi = gray[y:y + h, x:x + w]
    
    if w == 0 or h == 0:
        return region
    
    # Find content bounding box (non-white area)
    non_white = roi < white_thresh
    if np.sum(non_white) == 0:
        return region  # Entire region is white, can't trim
    
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return region
    
    y_start, y_end = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_start, x_end = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    
    # Add small padding to avoid cutting into content
    padding = 2
    x_start = max(0, x_start - padding)
    y_start = max(0, y_start - padding)
    x_end = min(w - 1, x_end + padding)
    y_end = min(h - 1, y_end + padding)
    
    # Only trim if we're removing a significant amount (>5% on any side)
    # Lowered threshold to be more aggressive about trimming margins
    trim_x = x_start > w * 0.05 or (w - x_end - 1) > w * 0.05
    trim_y = y_start > h * 0.05 or (h - y_end - 1) > h * 0.05
    
    if trim_x or trim_y:
        new_x = int(x + x_start)
        new_y = int(y + y_start)
        new_w = int(x_end - x_start + 1)
        new_h = int(y_end - y_start + 1)
        # Ensure minimum size
        if new_w >= 10 and new_h >= 10:
            return Region(new_x, new_y, new_w, new_h, region.label)
    
    return region


def _refine_region_boundaries(gray: np.ndarray, regions: List[Region]) -> List[Region]:
    """
    Post-process to refine all region boundaries using projection analysis.
    This tightens bounding boxes to better align with actual content boundaries.
    """
    refined = []
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    
    for region in regions:
        x, y, w, h = region.to_bbox()
        
        # Extract region from binary image
        # Slightly expand search area to find true boundaries
        expand = 5
        x_exp = max(0, x - expand)
        y_exp = max(0, y - expand)
        w_exp = min(binary.shape[1] - x_exp, w + 2 * expand)
        h_exp = min(binary.shape[0] - y_exp, h + 2 * expand)
        
        roi_binary = binary[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]
        
        if roi_binary.size == 0:
            refined.append(region)
            continue
        
        # Use projection analysis to find content boundaries
        proj_h = (255 - roi_binary).sum(axis=1).astype(np.float32)
        proj_v = (255 - roi_binary).sum(axis=0).astype(np.float32)
        
        if len(proj_h) > 0 and len(proj_v) > 0:
            # Normalize projections
            proj_h_norm = (proj_h - proj_h.min()) / (proj_h.max() - proj_h.min() + 1e-6)
            proj_v_norm = (proj_v - proj_v.min()) / (proj_v.max() - proj_v.min() + 1e-6)
            
            # Find content boundaries: where projection indicates content (not just whitespace)
            # Use a threshold that adapts to content type
            if region.label == "blank":
                # Blanks should be mostly white, so use higher threshold
                content_thresh = 0.15
            elif region.label == "table":
                # Tables have grid lines, so use moderate threshold
                content_thresh = 0.10
            else:
                # Text and figures: use lower threshold to catch all content
                content_thresh = 0.08
            
            h_content = np.where(proj_h_norm > content_thresh)[0]
            v_content = np.where(proj_v_norm > content_thresh)[0]
            
            if len(h_content) > 0 and len(v_content) > 0:
                # Find actual content boundaries
                y_start_refined = int(h_content[0])
                y_end_refined = int(h_content[-1])
                x_start_refined = int(v_content[0])
                x_end_refined = int(v_content[-1])
                
                # Add small padding to include content edges
                padding = 1
                y_start_refined = max(0, y_start_refined - padding)
                y_end_refined = min(h_exp - 1, y_end_refined + padding)
                x_start_refined = max(0, x_start_refined - padding)
                x_end_refined = min(w_exp - 1, x_end_refined + padding)
                
                # Convert back to image coordinates
                new_x = x_exp + x_start_refined
                new_y = y_exp + y_start_refined
                new_w = x_end_refined - x_start_refined + 1
                new_h = y_end_refined - y_start_refined + 1
                
                # Only use refined box if it's reasonable
                # Should be similar size to original (not too different)
                size_ratio = (new_w * new_h) / (w * h) if (w * h) > 0 else 0
                if (new_w >= 10 and new_h >= 10 and 
                    new_w < gray.shape[1] and new_h < gray.shape[0] and
                    0.2 < size_ratio < 2.5):  # Not too different in size
                    refined.append(Region(new_x, new_y, new_w, new_h, region.label))
                else:
                    refined.append(region)
            else:
                refined.append(region)
        else:
            refined.append(region)
    
    return refined


def _refine_table_boxes(gray: np.ndarray, regions: List[Region]) -> List[Region]:
    """
    Post-process to refine table bounding boxes.
    Tables often get over-segmented or have incorrect boundaries.
    This function tries to find the actual table boundaries by detecting grid lines.
    """
    refined = []
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    
    for region in regions:
        if region.label != "table":
            refined.append(region)
            continue
        
        x, y, w, h = region.to_bbox()
        
        # Expand search area to find table boundaries
        # Tables might be slightly larger than detected region
        expand = 15
        x_exp = max(0, x - expand)
        y_exp = max(0, y - expand)
        w_exp = min(binary.shape[1] - x_exp, w + 2 * expand)
        h_exp = min(binary.shape[0] - y_exp, h + 2 * expand)
        
        roi_binary = binary[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]
        
        if roi_binary.size == 0:
            refined.append(region)
            continue
        
        # Find table grid lines: look for horizontal and vertical lines
        # Tables have both horizontal (rows) and vertical (columns) lines
        proj_h = (255 - roi_binary).sum(axis=1).astype(np.float32)  # Horizontal projection
        proj_v = (255 - roi_binary).sum(axis=0).astype(np.float32)  # Vertical projection
        
        # Find where table content actually is (not just whitespace)
        # Use a threshold to find content regions
        if len(proj_h) > 0 and len(proj_v) > 0:
            # Normalize to find content vs whitespace
            proj_h_norm = (proj_h - proj_h.min()) / (proj_h.max() - proj_h.min() + 1e-6)
            proj_v_norm = (proj_v - proj_v.min()) / (proj_v.max() - proj_v.min() + 1e-6)
            
            # Find content boundaries: where projection is above threshold
            # Lower threshold to catch table lines (they're subtle)
            content_thresh = 0.05  # Lower threshold to catch table structure
            h_content = np.where(proj_h_norm > content_thresh)[0]
            v_content = np.where(proj_v_norm > content_thresh)[0]
            
            if len(h_content) > 0 and len(v_content) > 0:
                # Find actual table boundaries
                y_start_refined = int(h_content[0])
                y_end_refined = int(h_content[-1])
                x_start_refined = int(v_content[0])
                x_end_refined = int(v_content[-1])
                
                # Add small padding to include table borders
                padding = 2
                y_start_refined = max(0, y_start_refined - padding)
                y_end_refined = min(h_exp - 1, y_end_refined + padding)
                x_start_refined = max(0, x_start_refined - padding)
                x_end_refined = min(w_exp - 1, x_end_refined + padding)
                
                # Convert back to image coordinates
                new_x = x_exp + x_start_refined
                new_y = y_exp + y_start_refined
                new_w = x_end_refined - x_start_refined + 1
                new_h = y_end_refined - y_start_refined + 1
                
                # Only use refined box if it's reasonable
                # Should be similar size to original (not too different)
                size_ratio = (new_w * new_h) / (w * h) if (w * h) > 0 else 0
                if (new_w >= 15 and new_h >= 15 and 
                    new_w < gray.shape[1] and new_h < gray.shape[0] and
                    0.3 < size_ratio < 3.0):  # Not too different in size
                    refined.append(Region(new_x, new_y, new_w, new_h, region.label))
                else:
                    refined.append(region)
            else:
                refined.append(region)
        else:
            refined.append(region)
    
    return refined


def segment_image(image: np.ndarray, min_region: int = 40, max_depth: int = 12, do_merge: bool = True, use_feature_split: bool = False) -> List[Region]:
    """
    Segment an image into regions using XY-CUT (whitespace-based) approach.
    
    Args:
        image: Input image (BGR or grayscale)
        min_region: Minimum region size
        max_depth: Maximum recursion depth
        do_merge: Whether to perform greedy merging
        use_feature_split: If True, use feature-homogeneity/hybrid splitting (experimental).
                          If False (default), use whitespace-based XY-cut (recommended).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    # Binarize
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = thr.shape
    leaves = _xy_cut(thr, 0, 0, w, h, min_wh=min_region, depth=0, max_depth=max_depth, use_feature_split=use_feature_split)

    # Optional greedy merge of adjacent tiny regions with same label
    if do_merge:
        for r in leaves:
            r.label = _label_region(gray, r)
        leaves = _greedy_merge(leaves, iou_threshold=0.0, adjacency=3)
    else:
        for r in leaves:
            r.label = _label_region(gray, r)
    
    # Post-process: trim margins from each region to improve alignment
    trimmed = [_trim_margins(gray, r) for r in leaves]
    
    # Post-process: refine all region boundaries to better align with content
    # This improves bounding box accuracy for all content types
    refined = _refine_region_boundaries(gray, trimmed)
    
    # Additional refinement for tables (more aggressive)
    refined = _refine_table_boxes(gray, refined)
    
    return refined


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _are_adjacent(a: Region, b: Region, pad: int = 3) -> bool:
    ax0, ay0, aw, ah = a.to_bbox()
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = b.to_bbox()
    bx1, by1 = bx0 + bw, by0 + bh

    # Calculate actual gaps (not expanded)
    gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
    gap_y = max(0, max(ay0, by0) - min(ay1, by1))
    
    # Regions must overlap or be very close (within pad pixels)
    # But if there's a large gap in either direction, they're not adjacent
    if gap_x > pad or gap_y > pad:
        return False
    
    # If they overlap or are within pad pixels, check if they're actually touching
    # Expand by pad and test touches/overlaps
    ax0p, ay0p, ax1p, ay1p = ax0 - pad, ay0 - pad, ax1 + pad, ay1 + pad
    bx0p, by0p, bx1p, by1p = bx0 - pad, by0 - pad, bx1 + pad, by1 + pad
    inter_w = _overlap_1d(ax0p, ax1p, bx0p, bx1p)
    inter_h = _overlap_1d(ay0p, ay1p, by0p, by1p)
    
    # Must overlap in both dimensions (not just aligned in one)
    return inter_w > 0 and inter_h > 0


def _merge_boxes(a: Region, b: Region) -> Region:
    x0 = min(a.x, b.x)
    y0 = min(a.y, b.y)
    x1 = max(a.x + a.w, b.x + b.w)
    y1 = max(a.y + a.h, b.y + b.h)
    return Region(x0, y0, x1 - x0, y1 - y0, label=a.label)


def _greedy_merge(regions: List[Region], iou_threshold: float = 0.0, adjacency: int = 2) -> List[Region]:
    # Be very conservative: only merge regions that are truly adjacent and not separated
    merged = True
    regs = regions[:]
    while merged:
        merged = False
        out: List[Region] = []
        used = [False] * len(regs)
        for i in range(len(regs)):
            if used[i]:
                continue
            a = regs[i]
            acc = a
            for j in range(i + 1, len(regs)):
                if used[j]:
                    continue
                b = regs[j]
                # Only merge if same label
                if a.label != b.label:
                    continue
                
                # Check if regions are adjacent
                if not _are_adjacent(acc, b, pad=min(adjacency, 5)):
                    continue
                
                # Additional check: don't merge if there's another region between them
                # This prevents merging text blocks separated by figures/tables
                ax0, ay0, aw, ah = acc.to_bbox()
                ax1, ay1 = ax0 + aw, ay0 + ah
                bx0, by0, bw, bh = b.to_bbox()
                bx1, by1 = bx0 + bw, by0 + bh
                
                # Don't merge regions that are in different columns
                # This is critical for multi-column layouts
                # Check if they're in the same column (x overlap > 50% of smaller width)
                x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
                min_width = min(aw, bw)
                same_column = x_overlap > min_width * 0.5
                
                # Check if they're in the same row (y overlap > 50% of smaller height)
                y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
                min_height = min(ah, bh)
                same_row = y_overlap > min_height * 0.5
                
                # Calculate actual gap between regions
                gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
                gap_y = max(0, max(ay0, by0) - min(ay1, by1))
                
                # For tables: be VERY strict - only merge if they're in the same column/row AND very close
                # Tables in different columns should NEVER merge, even if touching
                if a.label == "table" and b.label == "table":
                    # CRITICAL: Tables must have significant overlap to merge (same column or same row)
                    # Just touching is not enough - they must actually overlap
                    if not (same_column or same_row):
                        continue  # Different columns/rows - NEVER merge, even if touching
                    # If they're side-by-side (no x overlap), don't merge
                    if x_overlap == 0 and gap_x == 0:
                        continue  # Side-by-side tables touching - don't merge
                    # If they're stacked (no y overlap), don't merge
                    if y_overlap == 0 and gap_y == 0:
                        continue  # Stacked tables touching - don't merge
                    # Only merge if they actually overlap AND are very close
                    if gap_x > 3 or gap_y > 3:
                        continue  # Too far apart - don't merge
                
                # For text regions, only merge if in same column (vertical merging)
                if a.label == "text":
                    if not same_column:
                        continue  # Don't merge text regions from different columns
                    # Also check gap - don't merge if there's a significant gap
                    if gap_y > 10:  # More than 10 pixels gap vertically
                        continue
                
                # For other labels (figures, blanks), be conservative
                if a.label != "text" and a.label != "table":
                    # Must be in same column or same row
                    if not (same_column or same_row):
                        continue
                    # Check gap size
                    if gap_x > 5 or gap_y > 5:
                        continue
                
                # Check if any other region overlaps the gap between acc and b
                has_obstacle = False
                for k, other in enumerate(regs):
                    if k == i or k == j or used[k]:
                        continue
                    ox0, oy0, ow, oh = other.to_bbox()
                    ox1, oy1 = ox0 + ow, oy0 + oh
                    
                    # Check if other region is in the gap between acc and b
                    gap_x0, gap_x1 = min(ax0, bx0), max(ax1, bx1)
                    gap_y0, gap_y1 = min(ay0, by0), max(ay1, by1)
                    
                    # If other region overlaps the gap area, it's an obstacle
                    if (ox0 < gap_x1 and ox1 > gap_x0 and 
                        oy0 < gap_y1 and oy1 > gap_y0):
                        has_obstacle = True
                        break
                
                if has_obstacle:
                    continue
                
                # Gap size already calculated above
                # Be very conservative: only merge if regions are touching or almost touching
                # This prevents merging blocks that are separated by gaps
                
                # Special handling for tables: never merge tables with other content types
                # Tables should remain isolated
                if a.label == "table" or b.label == "table":
                    # Only merge tables with other tables, and only if very close
                    if a.label == "table" and b.label == "table":
                        # Tables can merge if they're very close (same table split incorrectly)
                        # Gap check already done above
                        acc = _merge_boxes(acc, b)
                        used[j] = True
                        merged = True
                    continue  # Don't merge table with non-table
                
                # For other content types, merge if touching or very close
                max_gap = 3  # Allow slightly larger gap for non-table content
                if gap_x <= max_gap and gap_y <= max_gap:
                    acc = _merge_boxes(acc, b)
                    used[j] = True
                    merged = True
            used[i] = True
            out.append(acc)
        regs = out
    return regs


def draw_overlay(image: np.ndarray, regions: List[Region]) -> np.ndarray:
    color_map = {
        "text": (50, 180, 50),
        "table": (50, 50, 220),
        "figure": (220, 140, 40),
        "blank": (200, 200, 200),
        None: (180, 180, 180),
    }
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for r in regions:
        c = color_map.get(r.label, (150, 150, 150))
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), c, 2)
        cv2.putText(out, r.label or "region", (r.x + 3, r.y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--save_overlay", type=str, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--min_region", type=int, default=40)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--no_merge", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    regs = segment_image(img, min_region=args.min_region, max_depth=args.max_depth, do_merge=not args.no_merge)

    if args.save_overlay:
        overlay = draw_overlay(img, regs)
        os.makedirs(os.path.dirname(args.save_overlay), exist_ok=True)
        cv2.imwrite(args.save_overlay, overlay)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump([asdict(r) for r in regs], f, indent=2)


if __name__ == "__main__":
    main()


