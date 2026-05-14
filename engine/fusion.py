import cv2
import numpy as np


class FusionEngine:
    """v3 fusion with noise confirmation and multi-signal agreement."""

    @staticmethod
    def calculate_results(scores: dict) -> dict:
        noise = scores.get('noise', 0)

        # ── AI Generated ──
        exif_ai = scores.get('exif_ai', 0)
        jpeg_ghost = scores.get('jpeg_ghost', 0)
        deep = scores.get('deep', 0)

        # Base: EXIF is primary signal
        ai_generated = exif_ai

        # Confirmatory boost: elevated noise + no EXIF = strong AI signal
        if exif_ai >= 50 and noise > 18:
            ai_generated += min((noise - 18) * 1.5, 20)

        # Deep features boost (small, capped at 25)
        if exif_ai >= 30:
            ai_generated += deep * 0.4

        # JPEG ghost boost
        if jpeg_ghost > 10:
            ai_generated += jpeg_ghost * 0.3

        # ── Photoshop / Edited ──
        color_dist = scores.get('color_dist', 0)
        sat_contrast = scores.get('sat_contrast', 0)
        edit_sw = scores.get('edit_software', 0)
        ela = scores.get('ela', 0)
        ms_ela = scores.get('multiscale_ela', 0)
        noise_incon = scores.get('noise_inconsistency', 0)
        sharpening = scores.get('sharpening', 0)

        # Collect all edit signals
        all_edit = [
            ela * 4,           # ELA (boost because scores are typically 0-3)
            ms_ela * 4,        # Multi-scale ELA
            color_dist,        # Color grading / filters
            sat_contrast,      # Saturation / contrast manipulation
            noise_incon,       # Noise inconsistency (splicing)
            sharpening,        # Artificial sharpening
        ]

        # Primary: strongest signal drives the score
        top = max(all_edit)
        # Secondary: count how many signals are elevated (agreement)
        elevated = sum(1 for v in all_edit if v > 10)
        agreement_bonus = min(elevated * 5, 25)  # Up to +25 for multiple signals

        if edit_sw > 0:
            photoshop_edited = max(top, 45) + agreement_bonus
        else:
            photoshop_edited = top * 0.8 + agreement_bonus

        # ── Deepfake ──
        deepfake = scores.get('deepfake', 0)

        return {
            "ai_generated":     float(np.clip(ai_generated, 0, 100)),
            "photoshop_edited": float(np.clip(photoshop_edited, 0, 100)),
            "deepfake":         float(np.clip(deepfake, 0, 100)),
        }


def generate_heatmap(image_path, ela_img, noise_map=None, face_data=None, results=None):
    """
    Composite forensic heatmap with 3 layers:
      - ELA layer (cyan-blue): shows edited/re-saved regions
      - Noise layer (green-yellow): shows noise inconsistency (AI/splicing)
      - Face layer (red boxes): highlights deepfake face regions
    The dominant category gets the strongest visual weight.
    """
    original = cv2.imread(image_path)
    if original is None:
        return None

    h, w = original.shape[:2]

    # ── Layer 1: ELA (Edit Detection) ──
    if not isinstance(ela_img, np.ndarray):
        ela_img = cv2.cvtColor(np.array(ela_img), cv2.COLOR_RGB2BGR)
    ela_resized = cv2.resize(ela_img, (w, h))
    ela_gray = cv2.cvtColor(ela_resized, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ela_amp = np.clip(ela_gray * 3.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ela_enhanced = clahe.apply(ela_amp)
    # Gamma boost
    gamma = 0.6
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    ela_final = cv2.LUT(ela_enhanced, lut)
    ela_color = cv2.applyColorMap(ela_final, cv2.COLORMAP_JET)

    # ── Layer 2: Noise Map (AI Detection) ──
    if noise_map is not None:
        nm_resized = cv2.resize(noise_map, (w, h))
        nm_amp = np.clip(nm_resized.astype(np.float64) * 2.0, 0, 255).astype(np.uint8)
        nm_enhanced = clahe.apply(nm_amp)
        noise_color = cv2.applyColorMap(nm_enhanced, cv2.COLORMAP_HOT)
    else:
        noise_color = np.zeros_like(original)

    # ── Determine dominant category for blending weights ──
    if results:
        ai = results.get('ai_generated', 0)
        ed = results.get('photoshop_edited', 0)
        df = results.get('deepfake', 0)
        dominant = max(ai, ed, df)
    else:
        ai = ed = df = dominant = 0

    # Blending weights based on dominant result
    if dominant < 5:
        # Nothing detected — show subtle ELA
        w_ela, w_noise = 0.3, 0.1
    elif ed >= ai and ed >= df:
        w_ela, w_noise = 0.5, 0.1
    elif ai >= ed and ai >= df:
        w_ela, w_noise = 0.1, 0.5
    else:
        w_ela, w_noise = 0.2, 0.2

    w_orig = 1.0 - w_ela - w_noise

    # ── Composite blend ──
    composite = (original.astype(np.float64) * w_orig +
                 ela_color.astype(np.float64) * w_ela +
                 noise_color.astype(np.float64) * w_noise)
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # ── Layer 3: Deepfake face overlay ──
    if face_data:
        for (fx, fy, fw, fh, lap_var, face_score) in face_data:
            if face_score > 0:
                # Red box for suspicious faces
                color = (0, 0, 255) if face_score >= 50 else (0, 165, 255)
                thickness = 3 if face_score >= 50 else 2
                cv2.rectangle(composite, (fx, fy), (fx + fw, fy + fh), color, thickness)
                label = f"Deepfake: {face_score}%"
                cv2.putText(composite, label, (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Green box for normal faces
                cv2.rectangle(composite, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
                cv2.putText(composite, "Normal", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    return composite

