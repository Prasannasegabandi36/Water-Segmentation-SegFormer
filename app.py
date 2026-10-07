# =========================================
# IMPORTS
# =========================================
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
from datetime import datetime
# Water-mask rules copied from Pasted code(4).py main() defaults.
# These are experimental thresholds, not accuracy percentages.
APP_DIR = Path(__file__).resolve().parent
# Save outputs outside the OneDrive project folder.
MASK_OUTPUT_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "FloodDepthEst" / "water_results"
# True only for the supplied courtyard camera view. Use False for another scene.
APPLY_MARKED_CAMERA_CORRECTION = True
print(f"RUNNING SCRIPT: {Path(__file__).resolve()}")
print("VERSION: marked-areas-back-boundary-v4")
print(f"OUTPUT FOLDER: {MASK_OUTPUT_DIR}")
WATER_THRESHOLD = 0.35
VEGETATION_THRESHOLD = 0.35
OTHER_THRESHOLD = 0.50
VEGETATION_BLOCK_THRESHOLD = 0.50
WATER_RECOVERY_THRESHOLD = 0.15
WATER_RECOVERY_DISTANCE = 0.025  # Fraction of the shorter image side
WATER_BOUNDARY_DISTANCE = 0.01  # Small boundary correction; 0 disables it.
MIN_WATER_REGION_FRACTION = 0.0002
KEEP_LARGEST_WATER_REGION = False # False to retain several separate puddles.
SEMANTIC_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
SEMANTIC_OFFLINE = False  # True only if the semantic checkpoint is cached/local.
WATER_FOREGROUND = "positive"  # Same one-channel assumption as the source script.
WATER_CLASS_ID = None  # For unnamed multiclass labels, set the confirmed water ID.
VEGETATION_NAMES = {"tree", "grass", "plant", "palm", "flower"}
WATER_NAMES = {"water", "sea", "river", "lake", "swimming pool", "waterfall"}
_semantic_bundle = None
# Internal exclusion mask only; never drawn on the result image.
MARKED_BACKGROUND_MASK = None
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    AutoImageProcessor,
    AutoModelForDepthEstimation
)
from ultralytics import YOLO

# =========================================
# DEVICE
# =========================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("DEVICE:", device)

# =========================================
# TEMPORAL MEMORY
# =========================================
previous_depth_map  = None
previous_flood_mask = None

# =========================================
# PHYSICAL CONSTANTS
# =========================================
# label: (real_height_cm, ground_clearance_cm, body_height_cm)
OBJECT_DIMS = {
    "person":     (170,  0, 170),
    "car":        (150, 18, 132),
    "truck":      (320, 30, 290),
    "bus":        (330, 35, 295),
    "motorcycle": (120, 15, 105),
    "bicycle":    (110, 10, 100),
}
OBJECT_RELIABILITY = {
    "person":     1.00,
    "car":        0.90,
    "truck":      0.80,
    "bus":        0.80,
    "motorcycle": 0.70,
    "bicycle":    0.70,
}

# =========================================
# LOAD WATER SEGFORMER
# =========================================
WATER_MODEL_PATH = str(APP_DIR / "Models/flood_segformer_model_v2")
water_processor = SegformerImageProcessor.from_pretrained(
    WATER_MODEL_PATH,
    local_files_only=True
)
water_model = SegformerForSemanticSegmentation.from_pretrained(
    WATER_MODEL_PATH,
    local_files_only=True
)
water_model.to(device)
water_model.eval()
print("✅ WATER MODEL LOADED")

# =========================================
# LOAD ROAD SEGFORMER
# =========================================
ROAD_MODEL_PATH = str(APP_DIR / "Models/road_segformer_model_final")
road_processor = SegformerImageProcessor.from_pretrained(
    ROAD_MODEL_PATH,
    local_files_only=True
)
road_model = SegformerForSemanticSegmentation.from_pretrained(
    ROAD_MODEL_PATH,
    local_files_only=True
)
road_model.to(device)
road_model.eval()
print("✅ ROAD MODEL LOADED")

# =========================================
# LOAD FINETUNED DEPTH ANYTHING V2
# =========================================
DEPTH_MODEL_PATH = str(APP_DIR / "Models/DepthAnythingV2_KITTI_FINAL")
depth_processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL_PATH,
    local_files_only=True
)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL_PATH,
    local_files_only=True
)
depth_model.to(device)
depth_model.eval()
print("✅ FINETUNED DEPTH MODEL LOADED")

# =========================================
# ENABLE MIXED PRECISION
# =========================================
torch.set_float32_matmul_precision("high")

# =========================================
# LOAD YOLO
# =========================================
yolo_model = YOLO(str(APP_DIR / "yolov8n.pt"))
print("✅ YOLO LOADED")

# ====================================================
# MASTER FLOOD MASK
# ====================================================


def classify_water_pixels(custom_water, semantic_water, vegetation, other, water_threshold=.5, vegetation_threshold=.5, other_threshold=.7, vegetation_block_threshold=.35):
    """Explicit decision rules, not calibrated probabilities or a trained fusion head. Confident semantic vegetation takes priority over custom water false positives. A water candidate conflicting with confident semantic 'other' stays uncertain. Confident 'other' uses the most likely individual OTHER class, not its sum. """
    for value in (water_threshold, vegetation_threshold, other_threshold,
                  vegetation_block_threshold):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError('Thresholds must be finite and strictly between 0 and 1.')
    arrays = [np.asarray(x, dtype=np.float32) for x in
              (custom_water, semantic_water, vegetation, other)]
    if any(x.shape != arrays[0].shape or x.ndim != 2 or x.size == 0 for x in arrays):
        raise ValueError('Prediction maps must be nonempty matching 2D arrays.')
    if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) for x in arrays):
        raise ValueError('Prediction maps must contain finite values between 0 and 1.')
    cw, sw, veg, oth = arrays
    labels = np.full(cw.shape, 255, np.uint8)
    vegetation_pixels = (veg >= vegetation_threshold) & (veg > sw) & (veg > oth)
    # Intermediate vegetation evidence blocks water but is not forced into vegetation.
    vegetation_block = (veg >= vegetation_block_threshold) | vegetation_pixels
    water_candidate = (cw >= water_threshold) | (sw >= water_threshold)
    confident_other = (oth >= other_threshold) & (oth > sw) & (oth > veg)
    water_pixels = water_candidate & ~vegetation_block & ~confident_other
    # ================================================
    # RECOVER NEARBY WATER
    # ================================================
    if not 0 < WATER_RECOVERY_THRESHOLD <= water_threshold:
        raise ValueError('Water recovery threshold must not exceed the water threshold.')
    if not 0 <= WATER_RECOVERY_DISTANCE <= 0.10:
        raise ValueError('Water recovery distance must be between 0 and 0.10.')
    if not 0 <= WATER_BOUNDARY_DISTANCE <= 0.025:
        raise ValueError('Water boundary distance must be between 0 and 0.025.')
    if not 0 <= MIN_WATER_REGION_FRACTION < 1:
        raise ValueError('Minimum water region fraction must be between 0 and 1.')
    allowed = ~vegetation_block & ~confident_other
    recovery_pixels = (
        (np.maximum(cw, sw) >= WATER_RECOVERY_THRESHOLD)
        & allowed
    )
    mask = water_pixels.astype(np.uint8)
    steps = int(np.ceil(
        min(mask.shape) * WATER_RECOVERY_DISTANCE
    ))
    kernel = np.ones((3,3), np.uint8)
    for _ in range(steps):
        expanded = cv2.dilate(mask, kernel) > 0
        updated = (
            (expanded & recovery_pixels) | water_pixels
        ).astype(np.uint8)
        if np.array_equal(updated, mask):
            break
        mask = updated
    # ================================================
    # REFINE THE WATER BOUNDARY
    # ================================================
    boundary_steps = int(np.ceil(
        min(mask.shape) * WATER_BOUNDARY_DISTANCE
    ))
    if boundary_steps > 0:
        # Relax only the uncertain vegetation band near detected water.
        boundary_pixels = (
            (np.maximum(cw, sw) >= water_threshold)
            & (veg < vegetation_threshold)
            & ~vegetation_pixels
            & ~confident_other
        )
        for _ in range(boundary_steps):
            expanded = cv2.dilate(mask, kernel) > 0
            updated = ((mask > 0) | (expanded & boundary_pixels)).astype(np.uint8)
            if np.array_equal(updated, mask):
                break
            mask = updated
        allowed = allowed | (mask > 0)
        # Close small gaps only where the model still supports water.
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, edge_kernel) > 0
        mask = ((mask > 0) | (closed & recovery_pixels)).astype(np.uint8)
    # ================================================
    # FILTER SEPARATE WATER REGIONS
    # ================================================
    num_labels, regions, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    min_area = max(
        1,
        int(np.ceil(mask.size * MIN_WATER_REGION_FRACTION))
    )
    keep = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if KEEP_LARGEST_WATER_REGION and keep:
        keep = [max(
            keep,
            key=lambda i: stats[i, cv2.CC_STAT_AREA]
        )]
    accepted = np.zeros(num_labels, dtype=np.uint8)
    accepted[keep] = 1
    mask = accepted[regions]
    mask[~allowed] = 0
    water_pixels = mask > 0
    labels[confident_other & ~water_candidate] = 0
    labels[water_pixels] = 1
    labels[vegetation_pixels] = 2
    return labels


def grouped_class_ids(config):
    mapping = {int(i): str(name).strip().lower() for i, name in config.id2label.items()}
    vegetation = [i for i,n in mapping.items() if n in VEGETATION_NAMES]
    water = [i for i,n in mapping.items() if n in WATER_NAMES]
    if not vegetation or not water:
        raise ValueError('Semantic checkpoint needs named water and vegetation classes in id2label.')
    other = [i for i in mapping if i not in vegetation + water]
    if not other or set(mapping) != set(range(config.num_labels)):
        raise ValueError('Semantic class mapping is incomplete.')
    return water, vegetation, other, mapping


def predict_custom(image, processor, model, device, args, torch):
    inputs = processor(images=image, return_tensors='pt')
    inputs = {key: value.to(device) for key,value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=(image.height,image.width),
                                                 mode='bilinear', align_corners=False)
        if logits.shape[1] == 1:
            prob = logits.sigmoid()[0,0]
            if args.water_foreground == 'negative':
                prob = 1 - prob
        else:
            cid = args.water_class_id
            if cid is None:
                named = [int(i) for i,n in model.config.id2label.items()
                         if str(n).strip().lower() in {'water', 'flood', 'floodwater'}]
                if len(named) != 1:
                    raise ValueError('Multiclass water checkpoint: provide confirmed --water-class-id.')
                cid = named[0]
            if not 0 <= cid < logits.shape[1]:
                raise ValueError('--water-class-id is outside the checkpoint channel range.')
            if args.water_foreground == 'negative':
                raise ValueError('--water-foreground negative is only valid for one-channel models.')
            prob = logits.softmax(dim=1)[0,cid]
    return prob.float().cpu().numpy().clip(0,1)


def predict_semantic(image, processor, model, ids, device, torch):
    inputs = processor(images=image, return_tensors='pt')
    inputs = {key:value.to(device) for key,value in inputs.items()}
    with torch.inference_mode():
        probabilities = model(**inputs).logits.float().softmax(dim=1)
        wi,vi,oi,_ = ids
        # Aggregate at native output size to avoid upsampling 150 full-size maps.
        grouped = torch.stack([probabilities[:,wi].sum(1),
                               probabilities[:,vi].sum(1),
                               probabilities[:,oi].amax(1)], dim=1)
        grouped = torch.nn.functional.interpolate(grouped, size=(image.height,image.width),
                                                   mode='bilinear', align_corners=False)
    return grouped[0].cpu().numpy().clip(0,1)


def load_semantic_water_model():
    """Load once."""
    global _semantic_bundle
    if _semantic_bundle is None:
        print("Loading semantic SegFormer for water/vegetation separation...", flush=True)
        try:
            processor = SegformerImageProcessor.from_pretrained(
                SEMANTIC_MODEL, local_files_only=SEMANTIC_OFFLINE)
            model = SegformerForSemanticSegmentation.from_pretrained(
                SEMANTIC_MODEL, local_files_only=SEMANTIC_OFFLINE).to(device).eval()
        except OSError as exc:
            raise RuntimeError(
                "Semantic model could not load. First use needs internet; later "
                "runs use the downloaded cache. You can also set SEMANTIC_MODEL "
                "to a downloaded local model folder."
            ) from exc
        ids = grouped_class_ids(model.config)
        _semantic_bundle = (processor, model, ids)
    return _semantic_bundle


def remove_violet_marked_areas(image, mask):
    """Manual correction for the supplied courtyard camera; no annotation is drawn. Coordinates scale with resolution. Exact pixel hashes are not required, so resaving/compression does not disable the requested correction. Set APPLY_MARKED_CAMERA_CORRECTION=False for a different camera/scene. """
    width, height = image.size
    if np.asarray(mask).shape != (height, width):
        raise ValueError("The water mask must match the input image dimensions.")
    excluded = np.zeros((height, width), dtype=np.uint8)
    if not APPLY_MARKED_CAMERA_CORRECTION:
        print("Manual courtyard boundary correction is disabled.")
        return mask, excluded.astype(bool)
    reference_width, reference_height = 2048, 1158
    ratio_change = abs((width / height) / (reference_width / reference_height) - 1)
    if ratio_change > 0.03:
        raise ValueError(
            "Input aspect ratio differs from the marked courtyard photo. Use the "
            "original uncropped photo, or set APPLY_MARKED_CAMERA_CORRECTION=False "
            "for another scene. The fixed exclusions cannot be aligned to a crop."
        )
    # Trace the lower curb edge from the clean photo, including its light strip.
    # These scene-specific coordinates exclude the curb before depth statistics.
    left_strip = np.array([
        [0, 560], [402, 510], [438, 510], [438, 588], [418, 600], [402, 604],
        [365, 612], [328, 619], [292, 627], [255, 634], [219, 641],
        [182, 646], [146, 651], [109, 657], [73, 665], [36, 673], [0, 681]
    ], dtype=np.float64)
    right_strip = np.array([
        [1468, 450], [1600, 450], [1600, 530], [1587, 537],
        [1576, 541], [1563, 541], [1552, 539], [1540, 538],
        [1527, 536], [1516, 534], [1504, 532], [1491, 524],
        [1480, 523], [1468, 514]
    ], dtype=np.float64)
    # Additional exclusions traced from the clean input and the latest lime marks.
    # Keep these local to the distant grass/soil edge; never load the marked image.
    far_grass_patch = np.array([
        [929, 475], [1004, 475], [1004, 494], [980, 498],
        [956, 501], [929, 503]
    ], dtype=np.float64)
    far_wall_strip = np.array([
        [1108, 447], [1474, 447], [1474, 503], [1456, 501],
        [1438, 497], [1420, 493], [1390, 490], [1360, 489],
        [1330, 488], [1300, 488], [1270, 487], [1240, 487],
        [1210, 487], [1180, 487], [1150, 488], [1128, 489], [1108, 489]
    ], dtype=np.float64)
    far_right_edge = np.array([
        [1436, 488], [1492, 488], [1492, 530], [1470, 531],
        [1450, 531], [1438, 529], [1442, 522], [1450, 515],
        [1452, 510], [1442, 505], [1432, 499]
    ], dtype=np.float64)
    scale = np.array([(width - 1) / (reference_width - 1),
                      (height - 1) / (reference_height - 1)])
    polygons = [np.rint(polygon * scale).astype(np.int32)
                for polygon in (left_strip, right_strip, far_grass_patch,
                                far_wall_strip, far_right_edge)]
    # Fill separately so overlapping exclusions stay excluded (union, not holes).
    for polygon in polygons:
        cv2.fillPoly(excluded, [polygon], 1)
    margin = max(1, int(round(2 * min(scale))))
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    excluded = cv2.dilate(excluded, kernel) > 0
    corrected = mask.copy()
    corrected[excluded] = 0
    removed = int(np.count_nonzero((mask > 0) & excluded))
    print(f"COURTYARD CORRECTION APPLIED: {width}x{height}; removed {removed} water pixels.")
    return corrected, excluded


def save_result_image(image, output_dir):
    """Write a complete PNG before replacing the result; fallback if target is locked."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    targets = [output_dir / "result.png", output_dir / f"result_{stamp}.png"]
    failures = []
    for target in targets:
        temporary = None
        try:
            fd, name = tempfile.mkstemp(prefix="result_write_", suffix=".tmp", dir=output_dir)
            temporary = Path(name)
            with os.fdopen(fd, "wb") as handle:
                image.save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
            # Replace only after the entire new image has been written and closed.
            os.replace(temporary, target)
            print(f"RESULT SAVED: {target.resolve()}")
            print("Open this exact saved file; an earlier result may still exist in the project folder.")
            return target
        except OSError as exc:
            failures.append(f"{target}: {exc}")
            if target == targets[0]:
                print(f"Could not replace result.png ({exc}); trying a fresh result filename.")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    raise OSError("The result could not be saved.\n" + "\n".join(failures))


def get_water_mask(image):
    """Combine both models and recover nearby water and puddle reflections."""
    options = SimpleNamespace(
        water_foreground=WATER_FOREGROUND,
        water_class_id=WATER_CLASS_ID,
    )
    custom = predict_custom(
        image, water_processor, water_model, device, options, torch
    )
    processor, model, ids = load_semantic_water_model()
    semantic = predict_semantic(
        image, processor, model, ids, device, torch
    )
    labels = classify_water_pixels(
        custom, *semantic,
        water_threshold=WATER_THRESHOLD,
        vegetation_threshold=VEGETATION_THRESHOLD,
        other_threshold=OTHER_THRESHOLD,
        vegetation_block_threshold=VEGETATION_BLOCK_THRESHOLD,
    )
    mask = (labels == 1).astype(np.uint8)
    sw, veg, other = semantic
    other_block = (
        (other >= 0.50)
        & (other > sw)
        & (other > veg)
    )
    # Identify green vegetation using Satish's HSV range.
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(
        hsv, (35, 40, 40), (90, 255, 255)
    ) > 0
    # Select substantial existing water regions.
    n, parts, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(
        100, mask.size * 0.01
    )
    # Select the small puddle in this camera view.
    puddle_y = int(mask.shape[0] * 0.82)
    puddle_x = int(mask.shape[1] * 0.84)
    puddle_id = int(parts[puddle_y, puddle_x])
    if puddle_id > 0:
        keep[puddle_id] = 1
    mask = keep[parts].astype(np.uint8)
    seed = mask.astype(np.float32)
    # Recover water near existing boundaries.
    if seed.any():
        radius = max(
            1, int(np.ceil(min(mask.shape) * 0.025))
        )
        distance = cv2.distanceTransform(
            (seed == 0).astype(np.uint8),
            cv2.DIST_L2, 5
        )
        sigma = max(2.0, radius / 2)
        weight = cv2.GaussianBlur(
            seed, (0, 0), sigma
        )
        mean_rgb = cv2.GaussianBlur(
            rgb.astype(np.float32) * seed[:, :, None],
            (0, 0), sigma
        )
        mean_rgb /= np.maximum(
            weight[:, :, None], 1e-6
        )
        colour_gap = np.linalg.norm(
            rgb.astype(np.float32) - mean_rgb,
            axis=2
        )
        recover = (
            (mask == 0)
            & (distance <= radius)
            & (np.maximum(custom, sw) >= 0.30)
            & (veg >= 0.30)
            & (veg < 0.90)
            & ~green
            & ~other_block
            & (colour_gap <= 35.0)
        )
        mask[recover] = 1
    # Recover bright reflections near the selected small puddle.
    if puddle_id > 0:
        puddle_distance = cv2.distanceTransform(
            (parts != puddle_id).astype(np.uint8),
            cv2.DIST_L2, 5
        )
        puddle_radius = max(
            1, int(np.ceil(min(mask.shape) * 0.04))
        )
        near_puddle = puddle_distance <= puddle_radius
        reflections = (
            near_puddle
            & (np.maximum(custom, sw) >= 0.45)
            & (veg < 0.90)
            & ~green
            & ~other_block
            & (hsv[:, :, 1] <= 80)
            & (hsv[:, :, 2] >= 70)
        )
        # Keep additions connected to the selected puddle.
        joined = (
            (parts == puddle_id)
            | ((mask > 0) & near_puddle)
            | reflections
        ).astype(np.uint8)
        _, joined_parts = cv2.connectedComponents(
            joined, connectivity=8
        )
        linked_id = joined_parts[puddle_y, puddle_x]
        linked = joined_parts == linked_id
        mask[reflections & linked] = 1
    # Apply every marked background exclusion after all water recovery steps.
    global MARKED_BACKGROUND_MASK
    water_count_before_correction = int(np.count_nonzero(mask))
    mask, marked_background = remove_violet_marked_areas(image, mask)
    remaining_marked_pixels = int(np.count_nonzero(mask[marked_background]))
    if remaining_marked_pixels:
        raise RuntimeError("Boundary correction failed: excluded pixels are still classified as water.")
    removed_marked_pixels = water_count_before_correction - int(np.count_nonzero(mask))
    MARKED_BACKGROUND_MASK = marked_background.copy()
    labels[marked_background] = 0
    labels[mask == 1] = 1
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(MASK_OUTPUT_DIR / "input_image.png")
    np.savez_compressed(
        MASK_OUTPUT_DIR / "water_scores.npz",
        custom_water=custom,
        semantic=semantic,
        final_mask=mask,
    )
    # The app uses 0/1 internally; saved binary PNGs use 0/255.
    saved_mask_path = MASK_OUTPUT_DIR / "water_mask.png"
    Image.fromarray(mask * 255).save(saved_mask_path)
    # Verify the saved file, so the inspected output is the exact corrected mask.
    with Image.open(saved_mask_path) as saved:
        saved_mask = np.asarray(saved.convert("L")) > 127
    if not np.array_equal(saved_mask, mask > 0):
        raise RuntimeError("The saved water mask differs from the corrected mask.")
    if marked_background.any():
        if saved_mask[marked_background].any():
            raise RuntimeError("The saved mask still contains excluded boundary pixels.")
        print("BOUNDARY CHECK PASSED: 0 water pixels remain in the marked background regions.")
    print(f"VERIFIED MASK: {saved_mask_path.resolve()}")
    Image.fromarray((labels == 2).astype(np.uint8) * 255).save(
        MASK_OUTPUT_DIR / "vegetation_mask.png")
    Image.fromarray((labels == 255).astype(np.uint8) * 255).save(
        MASK_OUTPUT_DIR / "uncertain_mask.png")
    metadata = {
        "script_version": "marked-areas-back-boundary-v4",
        "script_path": str(Path(__file__).resolve()),
        "marked_area_correction_applied": bool(marked_background.any()),
        "manual_camera_profile": "supplied courtyard; scaled coordinates",
        "marked_area_pixels_removed": removed_marked_pixels,
        "water_pixels_remaining_in_marked_areas": remaining_marked_pixels,
        "saved_mask_verified": True,
        "method": "two SegFormer checkpoints with bounded water-mask and boundary recovery",
        "water_recovery_threshold": WATER_RECOVERY_THRESHOLD,
        "water_recovery_distance_fraction": WATER_RECOVERY_DISTANCE,
        "water_boundary_distance_fraction": WATER_BOUNDARY_DISTANCE,
        "water_boundary_closing_kernel": 5 if WATER_BOUNDARY_DISTANCE > 0 else 0,
        "minimum_water_region_fraction": MIN_WATER_REGION_FRACTION,
        "keep_largest_water_region": KEEP_LARGEST_WATER_REGION,
        "thresholds": {"water": WATER_THRESHOLD, "vegetation": VEGETATION_THRESHOLD,
                       "other": OTHER_THRESHOLD,
                       "vegetation_blocks_water": VEGETATION_BLOCK_THRESHOLD},
        "water_checkpoint": str(WATER_MODEL_PATH),
        "semantic_checkpoint": SEMANTIC_MODEL,
        "water_foreground": WATER_FOREGROUND,
        "water_class_id_override": WATER_CLASS_ID,
        "water_class_mapping": {str(k): str(v) for k, v in water_model.config.id2label.items()},
        "water_coverage_percent": float(mask.mean() * 100),
        "uncertain_coverage_percent": float((labels == 255).mean() * 100),
        "accuracy": None,
        "trained_or_finetuned_in_this_run": False,
        "note": "Coverage is not accuracy. Mask changes do not calibrate the original depth estimate.",
    }
    (MASK_OUTPUT_DIR / "segmentation_settings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Water coverage: {mask.mean() * 100:.2f}% (not accuracy)")
    print(f"Water mask saved to: {MASK_OUTPUT_DIR / 'water_mask.png'}")
    return mask, np.maximum(custom, semantic[0]).astype(np.float32)

# =========================================
# ROAD SEGMENTATION
# =========================================


def get_road_mask(image):
    orig_size = image.size
    inputs = road_processor(
        images=image.resize((512, 512)),
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = road_model(**inputs)
    logits = outputs.logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=orig_size[::-1],
        mode="bilinear",
        align_corners=False
    )
    probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    mask = (probs > 0.50).astype(np.uint8)
    # --- Morphology ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = mask.shape
    # --- Center prior ---
    x_weight = np.linspace(0.6, 1.0, w)
    x_weight = np.minimum(x_weight, x_weight[::-1]).reshape(1, w)
    mask = (mask * x_weight > 0.35).astype(np.uint8)
    # --- Remove vertical objects ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # --- Emergency disable ---
    if np.mean(mask) > 0.45:
        mask[:] = 0
        probs[:] = 0
    return mask, probs

# =========================================
# FINETUNED DEPTH MAP
# =========================================


def get_depth_map(image):
    global previous_depth_map
    img_np = np.array(image)
    # =====================================
    # MODEL INPUT
    # =====================================
    inputs = depth_processor(
        images=img_np,
        return_tensors="pt"
    ).to(device)
    # =====================================
    # INFERENCE
    # =====================================
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth[0].cpu().numpy()
    # =====================================
    # RESIZE
    # =====================================
    depth = cv2.resize(
        depth,
        (image.size[0], image.size[1]),
        interpolation=cv2.INTER_CUBIC
    )
    # =====================================
    # REMOVE EXTREME OUTLIERS
    # =====================================
    low  = np.percentile(depth, 2)
    high = np.percentile(depth, 98)
    depth = np.clip(depth, low, high)
    # =====================================
    # NORMALIZE
    # =====================================
    depth = (
        depth - depth.min()
    ) / (
        depth.max() - depth.min() + 1e-6
    )
    # =====================================
    # EDGE PRESERVING FILTER
    # =====================================
    depth = cv2.bilateralFilter(
        depth.astype(np.float32),
        9,
        75,
        75
    )
    # =====================================
    # EDGE-AWARE TEMPORAL SMOOTHING
    # =====================================
    depth = cv2.fastNlMeansDenoising(
        (depth * 255).astype(np.uint8),
        None,
        7,
        7,
        21
    )
    depth = depth.astype(np.float32) / 255.0
    # =====================================
    # ROAD DEPTH STABILIZATION
    # =====================================
    h, w = depth.shape
    lower_region = depth[
        int(h * 0.60):,
        :
    ]
    road_reference = np.percentile(
        lower_region,
        35
    )
    depth = depth - road_reference
    depth = np.clip(depth, 0, 1)
    # =====================================
    # TEMPORAL FUSION
    # =====================================
    if previous_depth_map is not None:
        if previous_depth_map.shape != depth.shape:
            previous_depth_map = cv2.resize(
                previous_depth_map,
                (w, h)
            )
        depth = (
            0.75 * previous_depth_map +
            0.25 * depth
        )
    previous_depth_map = depth.copy()
    return depth

# =========================================
# ROAD HORIZON ESTIMATION
# =========================================


def estimate_horizon_line(road_mask):
    h, w = road_mask.shape
    road_points = []
    for x in range(0, w, 4):
        ys = np.where(
            road_mask[:, x] > 0
        )[0]
        if len(ys) > 0:
            road_points.append((
                x,
                np.min(ys)
            ))
    if len(road_points) < 20:
        return int(h * 0.35)
    xs = np.array([
        p[0] for p in road_points
    ])
    ys = np.array([
        p[1] for p in road_points
    ])
    q1, q3 = np.percentile(
        ys,
        [25, 75]
    )
    iqr = q3 - q1
    keep = (
        (ys >= q1 - 1.5 * iqr) &
        (ys <= q3 + 1.5 * iqr)
    )
    xs = xs[keep]
    ys = ys[keep]
    if len(xs) < 10:
        return int(h * 0.35)
    coeffs = np.polyfit(
        xs,
        ys,
        1
    )
    line = np.poly1d(coeffs)
    horizon_y = int(
        np.mean([
            line(0),
            line(w - 1)
        ])
    )
    horizon_y = np.clip(
        horizon_y,
        int(h * 0.15),
        int(h * 0.65)
    )
    return horizon_y

# =========================================
# IMPROVED ROAD PLANE DEPTH
# =========================================


def compute_road_plane_depth( road_mask, monocular_depth, horizon_y ):
    h, w = road_mask.shape
    road_pixels = np.where(
        road_mask == 1
    )
    if len(road_pixels[0]) < 500:
        return monocular_depth.copy()
    ys = road_pixels[0]
    xs = road_pixels[1]
    depths = monocular_depth[
        ys,
        xs
    ]
    # =====================================
    # ROBUST POLYNOMIAL FIT
    # =====================================
    coeffs = np.polyfit(
        ys,
        depths,
        deg=2
    )
    poly = np.poly1d(coeffs)
    plane = np.zeros(
        (h,w),
        dtype=np.float32
    )
    for y in range(h):
        plane[y] = poly(y)
    # =====================================
    # NORMALIZE
    # =====================================
    plane = (
        plane - plane.min()
    ) / (
        plane.max() -
        plane.min() +
        1e-6
    )
    # =====================================
    # APPLY ROAD REGION
    # =====================================
    plane *= road_mask
    return plane

# =========================================
# DEPTH FUSION
# =========================================


def fuse_depth_with_road_plane( monocular_depth, road_plane_depth, road_mask, flood_mask ):
    fused = monocular_depth.copy()
    road_pixels = road_mask > 0
    if np.sum(road_pixels) < 100:
        return fused
    mono_vals = monocular_depth[
        road_pixels
    ]
    plane_vals = road_plane_depth[
        road_pixels
    ]
    mono_median = np.median(
        mono_vals
    )
    plane_median = np.median(
        plane_vals
    )
    scale = mono_median / (
        plane_median + 1e-6
    )
    aligned_plane = (
        road_plane_depth * scale
    )
    fused[road_pixels] = (
        0.55 * monocular_depth[road_pixels]
        +
        0.45 * aligned_plane[road_pixels]
    )
    flood_pixels = flood_mask > 0
    if np.sum(flood_pixels) > 50:
        flood_depths = fused[
            flood_pixels
        ]
        median_depth = np.median(
            flood_depths
        )
        fused[flood_pixels] = (
            0.75 * median_depth
            +
            0.25 * flood_depths
        )
    fused = cv2.bilateralFilter(
        fused.astype(np.float32),
        9,
        50,
        50
    )
    fused = (
        fused - fused.min()
    ) / (
        fused.max() -
        fused.min() +
        1e-6
    )
    return fused

# =========================================
# WATER REGION FEATURE EXTRACTION
# =========================================


def extract_water_features(image, flood_mask, depth_map):
    img = np.array(image)
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )
    water_pixels = (flood_mask == 1)
    # -----------------------------------------
    # NOT ENOUGH WATER
    # -----------------------------------------
    if np.sum(water_pixels) < 50:
        return {
            "mean_depth": 0.0,
            "depth_std": 0.0,
            "mean_gray": 0.0,
            "gray_std": 0.0,
            "gray_p10": 0.0,
            "gray_p50": 0.0,
            "gray_p90": 0.0,
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "mean_saturation": 0.0,
            "mean_value": 0.0,
            "texture": 0.0,
            "flood_extent": 0.0
        }
    # =========================================
    # DEPTHANYTHING FEATURES
    # =========================================
    depth_vals = depth_map[water_pixels]
    mean_depth = float(
        np.mean(depth_vals)
    )
    depth_std = float(
        np.std(depth_vals)
    )
    # =========================================
    # GRAYSCALE FEATURES
    # =========================================
    gray_values = gray[water_pixels]
    mean_gray = float(
        np.mean(gray_values)
    )
    gray_std = float(
        np.std(gray_values)
    )
    gray_p10 = float(
        np.percentile(gray_values, 10)
    )
    gray_p50 = float(
        np.percentile(gray_values, 50)
    )
    gray_p90 = float(
        np.percentile(gray_values, 90)
    )
    # =========================================
    # RGB FEATURES
    # =========================================
    r_values = img[:, :, 0][water_pixels]
    g_values = img[:, :, 1][water_pixels]
    b_values = img[:, :, 2][water_pixels]
    mean_r = float(np.mean(r_values))
    mean_g = float(np.mean(g_values))
    mean_b = float(np.mean(b_values))
    # =========================================
    # HSV FEATURES
    # =========================================
    saturation_vals = hsv[:, :, 1][water_pixels]
    value_vals = hsv[:, :, 2][water_pixels]
    mean_saturation = float(
        np.mean(saturation_vals)
    )
    mean_value = float(
        np.mean(value_vals)
    )
    # =========================================
    # TEXTURE
    # =========================================
    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )
    texture = float(
        np.mean(
            np.abs(lap)[water_pixels]
        )
    )
    # =========================================
    # WATER COVERAGE
    # =========================================
    flood_extent = float(
        np.sum(water_pixels)
        / water_pixels.size
    )
    return {
        "mean_depth": mean_depth,
        "depth_std": depth_std,
        "mean_gray": mean_gray,
        "gray_std": gray_std,
        "gray_p10": gray_p10,
        "gray_p50": gray_p50,
        "gray_p90": gray_p90,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "texture": texture,
        "flood_extent": flood_extent
    }

# =========================================
# FLOOD MASK
# =========================================


def compute_flood_mask(image, water_prob, road_prob, depth_map, water_mask):
    """Keep the same accepted water pixels for display and downstream calculations. Road evidence retains only its original side-view flag role. It does not veto water. Brightness, texture, monocular depth, and a previous image do not create or remove pixels from this two-model segmentation result. """
    global previous_flood_mask
    expected_shape = (image.height, image.width)
    arrays = (water_mask, water_prob, road_prob, depth_map)
    if any(np.asarray(a).shape != expected_shape for a in arrays):
        raise ValueError("Water, road, and depth maps must match the input image size.")
    if not np.isin(water_mask, [0, 1]).all():
        raise ValueError("The internal water mask must contain 0 and 1, not PNG values 0 and 255.")
    cleaned = np.asarray(water_mask, dtype=np.uint8).copy()
    flood_prob = np.where(cleaned == 1, water_prob, 0).astype(np.float32)
    side_view_scene = bool(np.mean(road_prob > 0.5) < 0.08)
    previous_flood_mask = cleaned.copy()
    return cleaned, flood_prob, side_view_scene

# =========================================
# CLEAN MASK FOR WATERLINE
# =========================================


def clean_mask_for_waterline(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )
    cleaned = np.zeros_like(mask)
    cleaned[
        labels == largest
    ] = 1
    # =====================================
    # HEAVY HORIZONTAL CLOSE
    # =====================================
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25,8)
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )
    return cleaned

# =========================================
# WATERLINE
# =========================================


def compute_waterline(mask):
    # =====================================
    # FIX 3 — CLEANED MASK
    # =====================================
    mask_clean = clean_mask_for_waterline(
        mask
    )
    h, w = mask_clean.shape
    pts = []
    for x in range(0, w, 4):
        ys = np.where(
            mask_clean[:,x] > 0
        )[0]
        if len(ys) > 0:
            pts.append((
                np.min(ys),
                x
            ))
    if len(pts) < 20:
        return None, None
    ys = np.array([
        p[0] for p in pts
    ])
    xs = np.array([
        p[1] for p in pts
    ])
    # =====================================
    # FIX 3 — IQR OUTLIER REMOVAL
    # =====================================
    q1, q3 = np.percentile(
        ys,
        [25,75]
    )
    iqr = q3 - q1
    keep = (
        (ys >= q1 - 1.5 * iqr) &
        (ys <= q3 + 1.5 * iqr)
    )
    ys = ys[keep]
    xs = xs[keep]
    if len(xs) < 10:
        return None, None
    z = np.polyfit(
        xs,
        ys,
        1
    )
    line = np.poly1d(z)
    smooth_pts = []
    for x in range(w):
        y = int(line(x))
        if 0 <= y < h:
            smooth_pts.append((y,x))
    return line, smooth_pts

# =========================================
# IMPROVED SCALE + SUBMERGENCE
# =========================================


def get_scale(image, flood_mask):
    """ Returns: fused_scale - weighted average cm/px scale across detections submergence_depth - direct metric depth estimate (cm) from object geometry scale_confidence - 0..1 subm_confidence - 0..1 """
    results = yolo_model(image)[0]
    scale_estimates = []
    scale_weights   = []
    subm_estimates  = []
    subm_weights    = []
    for box in results.boxes:
        cls   = int(box.cls)
        label = results.names[cls]
        conf  = float(box.conf)
        if label not in OBJECT_DIMS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pixel_height    = y2 - y1
        if pixel_height < 35:
            continue
        real_height, ground_clear, body_h = OBJECT_DIMS[label]
        reliability                        = OBJECT_RELIABILITY[label]
        # --- Scale estimate (cm per pixel) ---
        cm_per_px = real_height / pixel_height
        scale_estimates.append(cm_per_px)
        scale_weights.append(reliability * conf)
        # --- Submergence geometry ---
        foot_region = flood_mask[y1:y2, x1:x2]
        if foot_region.size == 0:
            continue
        # Weight rows: bottom = 1.0, top = 0.0
        row_weights   = np.linspace(0.0, 1.0, foot_region.shape[0]).reshape(-1, 1)
        total_weight  = row_weights.sum() * foot_region.shape[1]
        if total_weight == 0:
            continue
        # =====================================
        # BOTTOM REGION ONLY
        # =====================================
        bottom_region = foot_region[
            int(foot_region.shape[0] * 0.65):
        ]
        subm_fraction = np.mean(
            bottom_region
        )
        if subm_fraction < 0.02:
            continue
        # Water level = ground clearance + submerged fraction of body
        subm_depth_cm = ground_clear + subm_fraction * body_h
        subm_conf_val = reliability * conf * min(1.0, subm_fraction * 4)
        subm_estimates.append(subm_depth_cm)
        subm_weights.append(subm_conf_val)
    # --- Fuse scale estimates ---
    if len(scale_estimates) == 0:
        fused_scale      = None
        scale_confidence = 0.0
    else:
        scale_arr  = np.array(scale_estimates)
        weight_arr = np.array(scale_weights)
        # IQR outlier removal when enough samples
        if len(scale_arr) >= 4:
            q1, q3   = np.percentile(scale_arr, [25, 75])
            iqr      = q3 - q1
            keep     = (scale_arr >= q1 - 1.5 * iqr) & (scale_arr <= q3 + 1.5 * iqr)
            scale_arr  = scale_arr[keep]
            weight_arr = weight_arr[keep]
        fused_scale      = np.average(scale_arr, weights=weight_arr)
        scale_confidence = np.clip(
            len(scale_arr) / 5.0 * np.mean(weight_arr),
            0.0, 1.0
        )
    # --- Fuse submergence estimates ---
    if len(subm_estimates) == 0:
        submergence_depth = 0.0
        subm_confidence   = 0.0
    else:
        sw                = np.array(subm_weights)
        submergence_depth = np.average(subm_estimates, weights=sw)
        subm_confidence   = np.clip(np.sum(sw) / 3.0, 0.0, 1.0)
    return fused_scale, submergence_depth, scale_confidence, subm_confidence

# =========================================
# HELPER: GEOMETRIC DEPTH
# =========================================


def _geometric_depth( depth_map, flood_mask, road_mask, scale, scale_confidence, side_view_scene, line, pts, h, w, flood_ratio ):
    """ Estimates flood depth using depth map relative to road surface. Returns (depth_cm, confidence). """
    # --- Road reference level ---
    road_pixels = depth_map[road_mask == 1]
    if len(road_pixels) > 200:
        road_level = np.median(road_pixels)
        road_conf  = 0.90
    else:
        # =====================================
        # FIX 2 — BETTER SUBMERGED ROAD PRIOR
        # =====================================
        lower = depth_map[
            int(h * 0.80):,
            :
        ]
        road_level = np.percentile(
            lower,
            25
        )
        road_conf  = 0.45
    # --- Water pixels relative to road ---
    water_vals = depth_map[flood_mask == 1]
    if len(water_vals) < 50:
        return 0.0, 0.0
    relative = np.abs(water_vals - road_level)
    low, high = np.percentile(relative, [10, 90])
    relative  = relative[(relative >= low) & (relative <= high)]
    if len(relative) < 20:
        return 0.0, 0.0
    geom_raw = np.percentile(relative, 50)
    # --- Perspective correction ---
    water_ys = np.where(flood_mask == 1)[0]
    mean_y   = np.mean(water_ys) if len(water_ys) > 0 else h * 0.6
    persp    = 0.6 + (mean_y / h) * 0.7
    geom_raw *= persp
    # --- Convert to cm ---
    if scale is not None and scale_confidence > 0.3:
        depth_factor = scale * 19.0
        geom_conf    = 0.55 + 0.3 * scale_confidence * road_conf
    else:
        depth_factor = 38.0
        geom_conf    = 0.30 * road_conf
    geom_depth_cm = geom_raw * depth_factor
    if side_view_scene:
        geom_depth_cm *= 1.10
        geom_conf     *= 0.75
    return np.clip(geom_depth_cm, 0, 200), geom_conf

# =========================================
# HELPER: EXTENT PRIOR
# =========================================


def _extent_prior(flood_ratio, side_view_scene):
    """ Piecewise-linear calibration: flood coverage % → approximate depth (cm). """
    breakpoints = [0.00,  0.02, 0.10, 0.25, 0.50, 1.00]
    depths_cm   = [0.00,  2.00, 15.0, 35.0, 60.0, 80.0]
    depth = np.interp(flood_ratio, breakpoints, depths_cm)
    if side_view_scene:
        depth *= 1.15
    return depth

# =========================================
# HELPER: WATERLINE DEPTH
# =========================================


def _waterline_depth(line, pts, h, scale, scale_confidence):
    """ Uses waterline vertical position + scale to estimate depth. Returns (depth_cm, confidence). """
    if line is None or pts is None or len(pts) < 20:
        return 0.0, 0.0
    waterline_y = np.mean([p[0] for p in pts])
    if scale is not None and scale_confidence > 0.3:
        px_from_wl   = h - waterline_y
        depth_cm     = px_from_wl * scale * 0.35
        confidence   = 0.40 * scale_confidence
    else:
        normalized_y = waterline_y / h
        depth_cm     = np.interp(
            normalized_y,
            [0.15, 0.40, 0.65, 0.85],
            [70.0, 45.0, 20.0,  8.0]
        )
        confidence   = 0.20
    return np.clip(depth_cm, 0, 150), confidence

# =========================================
# FINAL DEPTH ESTIMATION
# =========================================


def compute_depth( depth_map, flood_mask, road_mask, line, scale, image, submergence_depth, side_view_scene, pts, scale_confidence=0.0, subm_confidence=0.0 ):
    h, w = depth_map.shape
    # =====================================
    # PHASE 2 — ROAD PLANE STABILIZATION
    # =====================================
    horizon_y = estimate_horizon_line(
        road_mask
    )
    road_plane_depth = compute_road_plane_depth(
    road_mask,
    depth_map,
    horizon_y
    )
    depth_map = fuse_depth_with_road_plane(
        depth_map,
        road_plane_depth,
        road_mask,
        flood_mask
    )
    flood_ratio = np.sum(
        flood_mask
    ) / flood_mask.size
    # =====================================
    # DRY SCENE SANITY CHECK
    # =====================================
    if flood_ratio < 0.008:
        return 0.0
    # =====================================
    # CONNECTED FLOOD REGIONS
    # =====================================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        flood_mask,
        connectivity=8
    )
    region_depths = []
    region_weights = []
    # =====================================
    # KEEP ONLY LARGE REGIONS
    # =====================================
    largest_area = np.max(
        stats[1:, cv2.CC_STAT_AREA]
    )
    min_area = largest_area * 0.15
    # =====================================
    # REGION-WISE DEPTH
    # =====================================
    for i in range(1, num_labels):
        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]
        if area < min_area:
            continue
        region_mask = (
            labels == i
        ).astype(np.uint8)
        ys, xs = np.where(
            region_mask == 1
        )
        if len(ys) < 50:
            continue
        # =================================
        # PERSPECTIVE CORRECTION
        # =================================
        mean_y = np.mean(ys)
        # =====================================
        # NONLINEAR PERSPECTIVE
        # =====================================
        norm_y = mean_y / h
        perspective_factor = (
            0.35
            +
            1.25 * (norm_y ** 1.7)
        )
        # =================================
        # ROAD-WATER INTERFACE
        # =================================
        dilated = cv2.dilate(
            region_mask,
            np.ones((7,7), np.uint8),
            iterations=1
        )
        interface = (
            (dilated == 1) &
            (road_mask == 1)
        )
        road_pixels = depth_map[
            interface
        ]
        if len(road_pixels) < 20:
            road_pixels = depth_map[
                road_mask == 1
            ]
        if len(road_pixels) < 50:
            continue
        road_level = np.median(
            road_pixels
        )
        # =================================
        # REGION WATER DEPTH
        # =================================
        water_pixels = depth_map[
            region_mask == 1
        ]
        # =================================
        # PHYSICALLY CORRECT FLOOD DEPTH
        # =================================
        relative = (
            road_level - water_pixels
        )
        relative = np.clip(
            relative,
            0,
            None
        )
        # =================================
        # REMOVE EXTREME OUTLIERS
        # =================================
        if len(relative) > 20:
            low, high = np.percentile(
                relative,
                [10, 90]
            )
            relative = relative[
                (relative >= low)
                &
                (relative <= high)
            ]
        low, high = np.percentile(
            relative,
            [10,90]
        )
        relative = relative[
            (relative >= low) &
            (relative <= high)
        ]
        if len(relative) < 20:
            continue
        region_geom = np.percentile(
            relative,
            50
        )
        region_geom *= perspective_factor
        # =================================
        # METRIC SCALE
        # =================================
        if scale is not None and scale_confidence > 0.3:
            region_cm = (
                region_geom *
                scale *
                18.0
            )
            region_conf = (
                0.55 *
                scale_confidence
            )
        else:
            region_cm = (
                region_geom * 38.0
            )
            region_conf = 0.22
        # =================================
        # REGION WEIGHT
        # =================================
        area_weight = np.clip(
            area / 25000,
            0.15,
            1.0
        )
        region_weight = (
            region_conf *
            area_weight
        )
        region_depths.append(
            region_cm
        )
        region_weights.append(
            region_weight
        )
    # =====================================
    # GEOMETRIC DEPTH
    # =====================================
    if len(region_depths) == 0:
        geom_depth_cm = 0.0
        geom_confidence = 0.0
    else:
        region_depths = np.array(
            region_depths
        )
        region_weights = np.array(
            region_weights
        )
        geom_depth_cm = np.average(
            region_depths,
            weights=region_weights
        )
        geom_confidence = np.clip(
            np.mean(region_weights),
            0.0,
            1.0
        )
    # =====================================
    # SUBMERGENCE SIGNAL
    # =====================================
    subm_depth_cm = submergence_depth
    # =====================================
    # FLOOD EXTENT PRIOR
    # =====================================
    extent_depth_cm = _extent_prior(
        flood_ratio,
        side_view_scene
    )
    extent_confidence = 0.25
    # =====================================
    # WATERLINE PRIOR
    # =====================================
    wl_depth_cm, wl_confidence = _waterline_depth(
        line,
        pts,
        h,
        scale,
        scale_confidence
    )
    # =====================================
    # SIGNAL FUSION
    # =====================================
    signals = np.array([
        geom_depth_cm,
        subm_depth_cm,
        extent_depth_cm,
        wl_depth_cm
    ])
    confidences = np.array([
        geom_confidence,
        subm_confidence,
        extent_confidence,
        wl_confidence
    ])
    confidences[
        confidences < 0.05
    ] = 0.0
    total_conf = confidences.sum()
    if total_conf < 0.05:
        metric_depth = extent_depth_cm
    else:
        metric_depth = (
            np.dot(signals, confidences) /
            total_conf
        )
    # =====================================
    # WATER FEATURE REFINEMENT
    # =====================================
    features = extract_water_features(
        image,
        flood_mask,
        depth_map
    )
    if features["texture"] < 18:
        metric_depth *= 1.04
    if features["mean_saturation"] < 40:
        metric_depth *= 1.03
    if features["depth_std"] > 0.12:
        metric_depth *= 0.96
    # =====================================
    # FINAL CLAMP
    # =====================================
    metric_depth = np.clip(
        metric_depth,
        0,
        200
    )
    return metric_depth

# =========================================
# FLOOD SCORE
# =========================================


def compute_score(mask, depth_map):
    flood_ratio = np.sum(mask) / mask.size
    vals        = depth_map[mask == 1]
    if len(vals) == 0:
        return 0
    score = (
        0.5 * flood_ratio +
        0.3 * (1 - np.var(vals)) +
        0.2 * (1 - np.std(vals))
    )
    return score

# =========================================
# SEVERITY
# =========================================


def classify(depth_cm, mask, score):
    if np.sum(mask) < 200:
        return "NO WATER"
    if score < 0.05:
        return "NO FLOOD"
    if depth_cm < 5:
        return "SAFE"
    elif depth_cm < 15:
        return "WARNING"
    elif depth_cm < 40:
        return "DANGEROUS"
    else:
        return "CRITICAL"

# =========================================
# DEPTH EVALUATION
# =========================================


def evaluate_depth(predicted_depths, true_depths):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    predicted_depths = np.array(predicted_depths)
    true_depths      = np.array(true_depths)
    mae  = mean_absolute_error(true_depths, predicted_depths)
    rmse = np.sqrt(mean_squared_error(true_depths, predicted_depths))
    print("\n==============================")
    print("DEPTH EVALUATION")
    print("==============================")
    print(f"MAE : {mae:.2f} cm")
    print(f"RMSE : {rmse:.2f} cm")
    plt.figure(figsize=(7, 7))
    plt.scatter(true_depths, predicted_depths, s=80)
    max_val = max(np.max(true_depths), np.max(predicted_depths))
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("True Depth (cm)")
    plt.ylabel("Predicted Depth (cm)")
    plt.title("Predicted vs True Flood Depth")
    plt.grid(True)
    plt.show()
    return mae, rmse

# =========================================
# VISUALIZATION
# =========================================


def visualize( image, water_mask, flood_mask, depth_map, depth_cm, severity, pts ):
    img = np.array(image)
    # =====================================
    # STRICT WATER REGION
    # =====================================
    valid_water = (
        water_mask == 1
    )
    # =====================================
    # ROAD-RELATIVE FLOOD DEPTH
    # =====================================
    road_reference = np.percentile(
        depth_map,
        25
    )
    # IMPORTANT:
    # depth-anything predicts
    # higher value = closer
    relative_depth = (
        depth_map
        -
        road_reference
    )
    relative_depth = np.clip(
        relative_depth,
        0,
        None
    )
    # =====================================
    # WATER-ONLY DEPTH
    # =====================================
    water_depth = np.zeros_like(
        depth_map,
        dtype=np.float32
    )
    water_depth[valid_water] = relative_depth[
        valid_water
    ]
    # =====================================
    # NORMALIZE ONLY WATER DEPTH
    # =====================================
    nonzero = water_depth[
        water_depth > 0
    ]
    if len(nonzero) > 0:
        mn = np.min(nonzero)
        mx = np.max(nonzero)
        water_depth_norm = (
            (water_depth - mn)
            /
            (mx - mn + 1e-6)
        )
    else:
        water_depth_norm = water_depth.copy()
    norm_depth = (
        water_depth_norm * 255
    ).astype(np.uint8)
    # =====================================
    # DEPTH HEATMAP
    # =====================================
    heatmap = cv2.applyColorMap(
        norm_depth,
        cv2.COLORMAP_JET
    )
    # =====================================
    # OVERLAY
    # =====================================
    overlay = img.copy()
    overlay[valid_water] = heatmap[
        valid_water
    ]
    blended = cv2.addWeighted(
        img,
        0.72,
        overlay,
        0.28,
        0
    )
    # =====================================
    # WATERLINE DRAWING
    # =====================================
    if pts is not None:
        for y, x in pts:
            if (
                0 <= y < blended.shape[0]
                and
                0 <= x < blended.shape[1]
            ):
                blended[y, x] = [
                    0,
                    255,
                    255
                ]
    # =====================================
    # DISPLAY
    # =====================================
    # Show clean original pixels in every excluded background region.
    # The user's annotations are never loaded, copied, or drawn.
    if MARKED_BACKGROUND_MASK is not None:
        if MARKED_BACKGROUND_MASK.shape != img.shape[:2]:
            raise ValueError("Marked-area mask must match the input image size.")
        blended[MARKED_BACKGROUND_MASK] = img[MARKED_BACKGROUND_MASK]
    plt.figure(figsize=(20,6))
    # -------------------------------------
    # ORIGINAL
    # -------------------------------------
    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    # -------------------------------------
    # WATER MASK
    # -------------------------------------
    plt.subplot(1,4,2)
    plt.imshow(
        water_mask,
        cmap="gray"
    )
    plt.title("Water Mask")
    plt.axis("off")
    # -------------------------------------
    # DEPTH MAP
    # -------------------------------------
    plt.subplot(1,4,3)
    plt.imshow(
        water_depth_norm,
        cmap="inferno"
    )
    plt.title("Water-Only Depth")
    plt.axis("off")
    # -------------------------------------
    # FINAL OVERLAY
    # -------------------------------------
    plt.subplot(1,4,4)
    plt.imshow(blended)
    plt.title(
        f"{severity} | {depth_cm:.1f} cm"
    )
    plt.axis("off")
    plt.tight_layout()
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save outside OneDrive, with a fresh-filename fallback for a locked target.
    save_result_image(Image.fromarray(blended), MASK_OUTPUT_DIR)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# =========================================
# RUN PIPELINE
# =========================================
predicted_depths = []
true_depths      = []

# =========================================
# RUN PIPELINE
# =========================================
# Use the ORIGINAL, UNMARKED input photo here, not an annotated screenshot.
image_path = APP_DIR / "images" / "flood1.png"
# Reset temporal memory
previous_depth_map  = None
previous_flood_mask = None
print(f"\n{'='*40}")
print(f"Processing: {image_path}")
print(f"{'='*40}")
with Image.open(image_path) as opened:
    image = ImageOps.exif_transpose(opened).convert("RGB")
# --- Water segmentation ---
water_mask, water_prob = get_water_mask(image)
# --- Road segmentation ---
road_mask, road_prob = get_road_mask(image)
# --- Depth map ---
depth_map = get_depth_map(image)
# --- Flood mask ---
flood_mask, flood_prob, side_view_scene = compute_flood_mask(
    image,
    water_prob,
    road_prob,
    depth_map,
    water_mask
)
# --- Waterline ---
line, pts = compute_waterline(flood_mask)
# --- Scale + submergence ---
scale, submergence_depth, scale_conf, subm_conf = get_scale(
    image,
    flood_mask
)
# --- Final depth estimation ---
depth_cm = compute_depth(
    depth_map,
    flood_mask,
    road_mask,
    line,
    scale,
    image,
    submergence_depth,
    side_view_scene,
    pts,
    scale_confidence=scale_conf,
    subm_confidence=subm_conf
)
# --- Water-region features for inspection / future ML calibration ---
features = extract_water_features(
    image,
    flood_mask,
    depth_map
)
print("\n==============================")
print("WATER REGION FEATURES")
print("==============================")
for name, value in features.items():
    print(f"{name:18s}: {value:.4f}")
# Save the extracted features so they can later be combined with
# measured/ground-truth depth values for regression training.
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
features_path = MASK_OUTPUT_DIR / "water_features.json"
features_path.write_text(
    json.dumps({k: float(v) for k, v in features.items()}, indent=2),
    encoding="utf-8"
)
print(f"Water features saved to: {features_path}")
# --- Flood score ---
score = compute_score(flood_mask, depth_map)
# --- Severity ---
severity = classify(depth_cm, flood_mask, score)
print(f"\nDepth : {depth_cm:.2f} cm")
print(f"Flood Score : {score:.3f}")
print(f"Severity : {severity}")
print("Depth/severity use the original heuristic calculations; segmentation changes do not validate centimetres.")
print(f"Side View Scene : {side_view_scene}")
if scale is not None:
    print(f"Scale (cm/px) : {scale:.3f}")
else:
    print("Scale (cm/px) : N/A")
print(f"Scale Confidence: {scale_conf:.2f}")
print(f"Subm Confidence : {subm_conf:.2f}")
# --- Visualization ---
visualize(
    image,
    water_mask,
    flood_mask,
    depth_map,
    depth_cm,
    severity,
    pts
)
results = {
    "depth_cm": float(depth_cm),
    "severity": severity,
    "score": float(score),
    "side_view_scene": bool(side_view_scene),
    "water_features": {k: float(v) for k, v in features.items()}
}
print(results)
