import numpy as np
from scipy.ndimage import distance_transform_edt
def compute_segmentation_metrics(pred_mask: np.ndarray,
                                  gt_mask: np.ndarray,
                                  spacing: tuple,
                                  nsd_tolerance_mm: float = 1.0) -> dict:
    """
    Shared evaluation function applied identically to U-Net, nnU-Net,
    and transformer outputs for a single anatomical structure.

    pred_mask, gt_mask : binary 3D numpy arrays (same shape)
    spacing            : voxel spacing in mm, e.g. (z, y, x)
    nsd_tolerance_mm    : clinically acceptable boundary tolerance
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # --- Overlap metrics ---
    intersection = np.logical_and(pred, gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-8)

    # --- Surface distance setup ---
    def surface_points(mask):
        eroded = distance_transform_edt(mask, sampling=spacing) > 0
        return mask & ~eroded  # boundary voxels

    pred_surf = surface_points(pred)
    gt_surf = surface_points(gt)

    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    pred_to_gt = dt_gt[pred_surf]
    gt_to_pred = dt_pred[gt_surf]

    # --- Boundary metrics ---
    hd95 = np.percentile(np.hstack([pred_to_gt, gt_to_pred]), 95)
    asd = np.mean(np.hstack([pred_to_gt, gt_to_pred]))

    # --- Clinical-tolerance boundary agreement (Normalized Surface Dice) ---
    nsd_num = (np.sum(pred_to_gt <= nsd_tolerance_mm) +
               np.sum(gt_to_pred <= nsd_tolerance_mm))
    nsd_den = len(pred_to_gt) + len(gt_to_pred) + 1e-8
    nsd = nsd_num / nsd_den

    return {
        "dice": dice,
        "iou": iou,
        "hd95_mm": hd95,
        "asd_mm": asd,
        "nsd": nsd,
    }
