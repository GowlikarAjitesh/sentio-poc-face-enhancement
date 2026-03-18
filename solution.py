import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ============================================================
# CS24M118 - Sentio POC Face Enhancement Assignment
# Student style implementation using only classical CV methods.
#
# This version carefully improves the previous pipeline while
# preserving the required order:
#   1) Denoising
#   2) CLAHE in LAB space
#   3) Multi-step Lanczos upscaling + sharpening
#   4) Zone sharpening
#
# Added careful improvements:
# - adaptive denoising
# - luminance-only sharpening
# - blockiness-aware smoothing
# - face-aware crop in final normalization
# - slightly safer contrast enhancement
# ============================================================

# -----------------------------
# Paths / constants
# -----------------------------
INPUT_DIR = "dataset/Profiles_1"
OUTPUT_DIR = "enhanced_faces"
REPORT_FILE = "enhancement_report.html"
METRICS_FILE = "evaluation_metrics.json"
TARGET_SIZE = (240, 240)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DENOISE_H = 8
DENOISE_H_COLOR = 8
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

CLAHE_CLIP_LIMIT = 3.5
CLAHE_TILE_GRID = (4, 4)

UPSCALE_THRESHOLD = 64

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -----------------------------
# File helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTS


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if is_image_file(name)
    ]


# -----------------------------
# Basic utilities
# -----------------------------
def compute_sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_ssim_same_size(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    img1_resized = cv2.resize(
        img1_bgr,
        (img2_bgr.shape[1], img2_bgr.shape[0]),
        interpolation=cv2.INTER_LANCZOS4
    )
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    return float(ssim(gray1, gray2, data_range=255))


def estimate_noise_level(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(noise))


def estimate_blockiness(img_bgr: np.ndarray) -> float:
    """
    Simple compression/blockiness estimate.
    I compare vertical and horizontal differences at 8-pixel boundaries.
    Larger value means stronger JPEG-like blocking.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    v_scores = []
    for x in range(8, w - 1, 8):
        boundary = np.mean(np.abs(gray[:, x] - gray[:, x - 1]))
        v_scores.append(boundary)

    h_scores = []
    for y in range(8, h - 1, 8):
        boundary = np.mean(np.abs(gray[y, :] - gray[y - 1, :]))
        h_scores.append(boundary)

    if not v_scores and not h_scores:
        return 0.0

    return float(np.mean(v_scores + h_scores))


def unsharp_mask_luminance(
    img_bgr: np.ndarray,
    amount: float = 0.8,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Sharpen only luminance channel for more natural result.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened_l = cv2.addWeighted(l_channel, 1.0 + amount, blurred, -amount, 0)

    sharpened_l = np.clip(sharpened_l, 0, 255).astype(np.uint8)
    merged = cv2.merge((sharpened_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def safe_blend(img1: np.ndarray, img2: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    blended = (
        img1.astype(np.float32) * alpha_mask[..., None] +
        img2.astype(np.float32) * (1.0 - alpha_mask[..., None])
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def detect_primary_face(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def pad_to_square_with_color(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)

    median_color = np.median(img_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
    canvas = np.full((side, side, 3), median_color, dtype=np.uint8)

    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y:y + h, x:x + w] = img_bgr
    return canvas


def face_centered_square_crop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Better final crop:
    - if face is detected, crop around face center
    - otherwise fall back to square padding
    """
    face = detect_primary_face(img_bgr)
    h, w = img_bgr.shape[:2]

    if face is None:
        return pad_to_square_with_color(img_bgr)

    x, y, fw, fh = face
    cx = x + fw // 2
    cy = y + fh // 2

    side = int(max(fw, fh) * 1.9)
    side = max(side, min(h, w))
    side = min(side, max(h, w))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.shape[0] != crop.shape[1]:
        crop = pad_to_square_with_color(crop)

    return crop


# -----------------------------
# Stage 1 - Denoising
# -----------------------------
def stage1_denoise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Adaptive denoising based on estimated noise.
    """
    noise_level = estimate_noise_level(img_bgr)

    if noise_level < 8:
        h_value = 5
        h_color_value = 5
    elif noise_level < 15:
        h_value = DENOISE_H
        h_color_value = DENOISE_H_COLOR
    else:
        h_value = 11
        h_color_value = 11

    out = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h_value,
        h_color_value,
        DENOISE_TEMPLATE_WINDOW,
        DENOISE_SEARCH_WINDOW,
    )

    return out


# -----------------------------
# Stage 2 - CLAHE on L channel only
# -----------------------------
def stage2_clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Slightly adaptive CLAHE:
    If image is already very contrasty, reduce clipLimit a little.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    std_l = float(np.std(l_channel))
    if std_l > 55:
        clip_limit = 2.8
    else:
        clip_limit = CLAHE_CLIP_LIMIT

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=CLAHE_TILE_GRID,
    )
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# -----------------------------
# Stage 3 - Multi-step Lanczos upscaling
# -----------------------------
def stage3_multistep_upscale(img_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-step upscaling with gentle edge-preserving cleanup.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    blockiness = estimate_blockiness(out)
    if blockiness > 20:
        out = cv2.medianBlur(out, 3)

    if min(h, w) < UPSCALE_THRESHOLD:
        out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # If image is very blocky, sharpen less.
        if blockiness > 18:
            out = unsharp_mask_luminance(out, amount=0.35, sigma=1.0)
        else:
            out = unsharp_mask_luminance(out, amount=0.50, sigma=1.0)

        h2, w2 = out.shape[:2]
        out = cv2.resize(out, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)

        if blockiness > 18:
            out = unsharp_mask_luminance(out, amount=0.20, sigma=1.0)
        else:
            out = unsharp_mask_luminance(out, amount=0.30, sigma=1.0)

    if out.shape[0] < TARGET_SIZE[1] or out.shape[1] < TARGET_SIZE[0]:
        out = cv2.resize(out, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Gentle bilateral cleanup after upscaling to reduce blocky edges
    out = cv2.bilateralFilter(out, d=3, sigmaColor=12, sigmaSpace=12)

    return out


# -----------------------------
# Stage 4 - Zone sharpening
# -----------------------------
def build_zone_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    face = detect_primary_face(img_bgr)

    if face is None:
        center_x = w // 2
        center_y = int(h * 0.45)
        axes = (int(w * 0.22), int(h * 0.20))
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
    else:
        x, y, fw, fh = face

        eye_nose_center = (x + fw // 2, y + int(fh * 0.42))
        eye_nose_axes = (max(12, int(fw * 0.22)), max(10, int(fh * 0.18)))
        cv2.ellipse(mask, eye_nose_center, eye_nose_axes, 0, 0, 360, 1.0, -1)

        mouth_center = (x + fw // 2, y + int(fh * 0.70))
        mouth_axes = (max(10, int(fw * 0.16)), max(8, int(fh * 0.10)))
        cv2.ellipse(mask, mouth_center, mouth_axes, 0, 0, 360, 0.55, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return np.clip(mask, 0.0, 1.0)


def stage4_zone_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    """
    Careful final sharpening.
    On blocky images, keep it softer.
    """
    mask = build_zone_mask(img_bgr)
    blockiness = estimate_blockiness(img_bgr)

    if blockiness > 18:
        strong = unsharp_mask_luminance(img_bgr, amount=0.60, sigma=1.0)
        mild = unsharp_mask_luminance(img_bgr, amount=0.12, sigma=1.2)
    else:
        strong = unsharp_mask_luminance(img_bgr, amount=0.92, sigma=0.95)
        mild = unsharp_mask_luminance(img_bgr, amount=0.22, sigma=1.15)

    return safe_blend(strong, mild, mask)


# -----------------------------
# Final normalization
# -----------------------------
def normalize_to_240(img_bgr: np.ndarray) -> np.ndarray:
    square = face_centered_square_crop(img_bgr)
    final_img = cv2.resize(square, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return final_img


# -----------------------------
# Full pipeline
# -----------------------------
def enhance_face(img_bgr: np.ndarray) -> np.ndarray:
    out = stage1_denoise(img_bgr)
    out = stage2_clahe_lab(out)
    out = stage3_multistep_upscale(out)
    out = stage4_zone_sharpen(out)
    out = normalize_to_240(out)
    return out


# -----------------------------
# HTML helpers
# -----------------------------
def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def relative_image_path(path: str) -> str:
    return path.replace("\\", "/")


# -----------------------------
# Report generation
# -----------------------------
def generate_html_report(rows: List[Dict], summary: Dict) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    html_parts.append("<title>Face Enhancement Report</title>")
    html_parts.append(
        """
        <style>
            body {
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                background: #f5f7fb;
                color: #1f2937;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 24px;
            }
            .header {
                background: #111827;
                color: white;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
            }
            .header h1 {
                margin: 0 0 10px 0;
                font-size: 28px;
            }
            .muted {
                color: #cbd5e1;
                font-size: 14px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .card {
                background: white;
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.06);
            }
            .card h3 {
                margin: 0 0 8px 0;
                font-size: 14px;
                color: #475569;
            }
            .value {
                font-size: 26px;
                font-weight: bold;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 14px;
                overflow: hidden;
                box-shadow: 0 8px 20px rgba(0,0,0,0.06);
            }
            th, td {
                border-bottom: 1px solid #e5e7eb;
                padding: 14px;
                text-align: center;
                vertical-align: middle;
                font-size: 14px;
            }
            th {
                background: #e2e8f0;
                color: #0f172a;
            }
            tr:hover {
                background: #f8fafc;
            }
            img {
                width: 140px;
                height: 140px;
                object-fit: cover;
                border-radius: 10px;
                border: 1px solid #cbd5e1;
                background: #e5e7eb;
            }
            .note {
                background: #fff7ed;
                border-left: 4px solid #f97316;
                padding: 14px;
                border-radius: 10px;
                margin: 18px 0 24px 0;
                color: #7c2d12;
            }
        </style>
        """
    )
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<div class='container'>")
    html_parts.append("<div class='header'>")
    html_parts.append("<h1>Low-Resolution CCTV Face Enhancement Report</h1>")
    html_parts.append("<div class='muted'>Generated offline on: " + html_escape(generated_at) + "</div>")
    html_parts.append("<div class='muted'>Student submission report with A/B comparison and quality metrics.</div>")
    html_parts.append("</div>")

    html_parts.append("<div class='grid'>")
    html_parts.append(f"<div class='card'><h3>Total Faces</h3><div class='value'>{summary['total_faces']}</div></div>")
    html_parts.append(f"<div class='card'><h3>Avg Sharpness (Raw)</h3><div class='value'>{summary['avg_sharpness_raw']:.2f}</div></div>")
    html_parts.append(f"<div class='card'><h3>Avg Sharpness (Enhanced)</h3><div class='value'>{summary['avg_sharpness_enhanced']:.2f}</div></div>")
    html_parts.append(f"<div class='card'><h3>Avg Sharpness Gain</h3><div class='value'>{summary['avg_sharpness_gain']:.2f}</div></div>")
    html_parts.append(f"<div class='card'><h3>Avg SSIM</h3><div class='value'>{summary['avg_ssim']:.4f}</div></div>")
    html_parts.append(f"<div class='card'><h3>Avg Runtime / Face</h3><div class='value'>{summary['avg_runtime_ms']:.2f} ms</div></div>")
    html_parts.append("</div>")

    html_parts.append(
        "<div class='note'><strong>Important note:</strong> The official assignment mentions a separate <code>reference_identities/</code> folder to compute recognition accuracy improvement. That folder is not present in the provided dataset bundle here. So this report measures the enhancement quality honestly using available metrics only: sharpness gain, SSIM, and runtime.</div>"
    )

    html_parts.append("<table>")
    html_parts.append("<tr><th>Filename</th><th>Raw</th><th>Enhanced</th><th>Raw Sharpness</th><th>Enhanced Sharpness</th><th>Gain</th><th>SSIM</th><th>Runtime</th></tr>")

    for row in rows:
        html_parts.append(
            "<tr>"
            f"<td>{html_escape(row['filename'])}</td>"
            f"<td><img src='{html_escape(relative_image_path(row['raw_rel']))}' alt='raw'></td>"
            f"<td><img src='{html_escape(relative_image_path(row['enh_rel']))}' alt='enhanced'></td>"
            f"<td>{row['sharpness_raw']:.2f}</td>"
            f"<td>{row['sharpness_enhanced']:.2f}</td>"
            f"<td>{row['sharpness_gain']:.2f}</td>"
            f"<td>{row['ssim']:.4f}</td>"
            f"<td>{row['runtime_ms']:.2f} ms</td>"
            "</tr>"
        )

    html_parts.append("</table>")
    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


# -----------------------------
# Main processing
# -----------------------------
def main() -> None:
    ensure_dir(OUTPUT_DIR)
    image_paths = list_images(INPUT_DIR)

    if not image_paths:
        raise FileNotFoundError(
            f"No input images found inside '{INPUT_DIR}'. Please place dataset images there first."
        )

    per_face_rows = []
    sharp_raw_sum = 0.0
    sharp_enh_sum = 0.0
    ssim_sum = 0.0
    runtime_sum = 0.0

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if raw is None:
            print(f"[WARNING] Could not read image: {filename}")
            continue

        start = time.perf_counter()
        enhanced = enhance_face(raw)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        out_name = os.path.splitext(filename)[0] + ".jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        cv2.imwrite(out_path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        sharp_raw = compute_sharpness(raw)
        sharp_enh = compute_sharpness(enhanced)
        ssim_value = compute_ssim_same_size(raw, enhanced)

        sharp_raw_sum += sharp_raw
        sharp_enh_sum += sharp_enh
        ssim_sum += ssim_value
        runtime_sum += runtime_ms

        per_face_rows.append({
            "filename": filename,
            "raw_rel": image_path,
            "enh_rel": out_path,
            "sharpness_raw": sharp_raw,
            "sharpness_enhanced": sharp_enh,
            "sharpness_gain": sharp_enh - sharp_raw,
            "ssim": ssim_value,
            "runtime_ms": runtime_ms,
        })

    total = len(per_face_rows)
    if total == 0:
        raise RuntimeError("No valid images were processed.")

    summary = {
        "dataset_used": INPUT_DIR,
        "total_faces": total,
        "target_output_size": {
            "width": TARGET_SIZE[0],
            "height": TARGET_SIZE[1]
        },
        "pipeline": [
            "Stage 1 - Adaptive denoising using cv2.fastNlMeansDenoisingColored",
            "Stage 2 - CLAHE in LAB space on L channel only",
            "Stage 3 - Multi-step Lanczos upscaling with luminance sharpening and gentle bilateral cleanup",
            "Stage 4 - Face-aware zone sharpening stronger around eye+nose region",
        ],
        "avg_sharpness_raw": sharp_raw_sum / total,
        "avg_sharpness_enhanced": sharp_enh_sum / total,
        "avg_sharpness_gain": (sharp_enh_sum - sharp_raw_sum) / total,
        "avg_ssim": ssim_sum / total,
        "avg_runtime_ms": runtime_sum / total,
        "recognition_accuracy_improvement": None,
        "recognition_accuracy_note": "Not computed because separate reference_identities/ folder was not available in the provided dataset bundle.",
        "faces": [
            {
                "filename": row["filename"],
                "sharpness_raw": row["sharpness_raw"],
                "sharpness_enhanced": row["sharpness_enhanced"],
                "sharpness_gain": row["sharpness_gain"],
                "ssim": row["ssim"],
                "runtime_ms": row["runtime_ms"],
                "enhanced_output": relative_image_path(row["enh_rel"]),
            }
            for row in per_face_rows
        ],
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    generate_html_report(per_face_rows, summary)

    print("\n=== Assignment Completed Successfully ===")
    print(f"Processed faces           : {total}")
    print(f"Enhanced output folder   : {OUTPUT_DIR}")
    print(f"HTML report              : {REPORT_FILE}")
    print(f"Metrics JSON             : {METRICS_FILE}")
    print(f"Average sharpness (raw)  : {summary['avg_sharpness_raw']:.4f}")
    print(f"Average sharpness (enh.) : {summary['avg_sharpness_enhanced']:.4f}")
    print(f"Average sharpness gain   : {summary['avg_sharpness_gain']:.4f}")
    print(f"Average SSIM             : {summary['avg_ssim']:.4f}")
    print(f"Average runtime / face   : {summary['avg_runtime_ms']:.2f} ms")


if __name__ == "__main__":
    main()