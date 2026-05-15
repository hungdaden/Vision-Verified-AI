import cv2
import numpy as np
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
import os
import torch
import torch.nn as nn
from torchvision import models, transforms


class ForensicEngine:
    """Production forensics engine v3 — proven methods + smart confirmatory signals."""

    _resnet = None
    _device = None

    # ═══════════════════════════════════════════
    #  EDIT DETECTION
    # ═══════════════════════════════════════════

    @staticmethod
    def run_ela(image_path, quality=90, multiplier=15):
        original = Image.open(image_path).convert('RGB')
        tmp = os.path.join(os.path.dirname(image_path) or '.', '_tmp_ela.jpg')
        original.save(tmp, 'JPEG', quality=quality)
        resaved = Image.open(tmp)
        diff = ImageChops.difference(original, resaved)
        enhanced = ImageChops.multiply(
            diff, Image.new('RGB', diff.size, (multiplier,) * 3))
        if os.path.exists(tmp):
            os.remove(tmp)
        score = np.mean(np.array(diff, dtype=np.float32)) / 255.0 * 100
        return enhanced, score

    @staticmethod
    def run_multiscale_ela(image_path):
        scores = [ForensicEngine.run_ela(image_path, quality=q)[1] for q in [95, 85, 75]]
        return min(np.var(scores) * 10, 100)

    @staticmethod
    def run_noise_analysis(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = cv2.absdiff(gray, cv2.medianBlur(gray, 5))
        score = np.std(noise) / 25.5 * 100
        return cv2.applyColorMap(noise, cv2.COLORMAP_JET), min(score, 100)

    # ═══════════════════════════════════════════
    #  FILTER / COLOR DETECTION
    # ═══════════════════════════════════════════

    @staticmethod
    def _hist_gap_ratio(hist, lo=10, hi=246):
        """Compute gap ratio in histogram mid-range. Curves/levels create gaps."""
        mid = hist[lo:hi]
        total = mid.sum()
        if total < 100:
            return 0.0
        nz = np.where(mid > total * 0.00005)[0]
        if len(nz) < 15:
            return 0.0
        span = nz[-1] - nz[0] + 1
        if span < 25:
            return 0.0
        populated = np.sum(mid[nz[0]:nz[-1] + 1] > 0)
        return 1.0 - (populated / span)

    @staticmethod
    def run_color_distribution(image_path):
        """
        Detects color manipulation: curves, levels, color grading, filters.
        Analyses histogram gaps, roughness, and channel correlations in
        both RGB and HSV spaces.
        """
        img = cv2.imread(image_path)
        if img is None:
            return 0
        score = 0.0
        total_gap = 0.0
        total_roughness = 0.0

        # ── RGB histogram analysis ──
        for ch in range(3):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256]).flatten()
            total_px = hist.sum()
            hist_n = hist / (total_px + 1e-8)

            # Clipping detection (lowered threshold)
            clip_lo = hist_n[:4].sum()
            clip_hi = hist_n[252:].sum()
            if clip_lo > 0.05:
                score += (clip_lo - 0.05) * 250
            if clip_hi > 0.05:
                score += (clip_hi - 0.05) * 250

            # Narrow dynamic range
            nz = np.where(hist > total_px * 0.0003)[0]
            if len(nz) > 0 and (nz[-1] - nz[0]) < 150:
                score += (150 - (nz[-1] - nz[0])) * 0.15

            # Histogram gap/comb detection
            total_gap += ForensicEngine._hist_gap_ratio(hist)

            # Histogram roughness (jaggedness from value remapping)
            kernel = np.ones(7) / 7
            smoothed = np.convolve(hist_n, kernel, mode='same')
            roughness = np.mean(np.abs(hist_n[15:240] - smoothed[15:240]))
            total_roughness += roughness

        # ── HSV histogram gap analysis ──
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for ch_idx in [1, 2]:  # Saturation and Value channels
            h_hist = cv2.calcHist([hsv], [ch_idx], None, [256], [0, 256]).flatten()
            gap = ForensicEngine._hist_gap_ratio(h_hist, lo=5, hi=250)
            if gap > 0.02:
                total_gap += gap * 1.5  # HSV gaps weigh more for color edits

        # Gap scoring
        if total_gap > 0.02:
            score += min(total_gap * 180, 50)

        # Roughness scoring
        if total_roughness > 0.004:
            score += min((total_roughness - 0.004) * 2500, 20)

        # ── Channel correlation ──
        # Natural photos: R,G,B are highly correlated
        # Color grading breaks this correlation
        b, g, r = cv2.split(img)
        step = max(1, img.shape[0] * img.shape[1] // 40000)
        r_s = r.flatten()[::step].astype(np.float64)
        g_s = g.flatten()[::step].astype(np.float64)
        b_s = b.flatten()[::step].astype(np.float64)

        def _safe_corr(a, b_arr):
            if np.std(a) < 1 or np.std(b_arr) < 1:
                return 1.0
            return np.corrcoef(a, b_arr)[0, 1]

        corrs = [_safe_corr(r_s, g_s), _safe_corr(r_s, b_s), _safe_corr(g_s, b_s)]
        min_corr = min(corrs)
        if min_corr < 0.78:
            score += (0.78 - min_corr) * 100

        return float(np.clip(score, 0, 100))

    @staticmethod
    def run_saturation_contrast(image_path):
        """
        Detects saturation/contrast manipulation via HSV statistics,
        distribution shape analysis, and CDF linearity testing.
        """
        img = cv2.imread(image_path)
        if img is None:
            return 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float64)
        v = hsv[:, :, 2].astype(np.float64)
        score = 0.0

        # ── Saturation analysis ──
        ms = np.mean(s)
        if ms > 115:
            score += (ms - 115) * 0.6
        elif ms < 15:
            score += (15 - ms) * 2

        # Saturation kurtosis: edited images tend to have peakier distribution
        s_std = np.std(s)
        if s_std > 1:
            s_kurt = float(np.mean(((s - ms) / s_std) ** 4) - 3.0)
            if s_kurt > 3.0:
                score += min((s_kurt - 3.0) * 3, 20)

        # Saturation uniformity: low std relative to mean = artificial boost
        if ms > 60 and s_std < ms * 0.25:
            score += min((ms * 0.25 - s_std) * 0.5, 15)

        # ── Brightness / contrast analysis ──
        bright = np.mean(v > 220)
        dark = np.mean(v < 30)
        if bright > 0.20 and dark > 0.20:
            score += 20

        # Center vs edge brightness (vignette detection)
        h, w = v.shape
        ctr = np.mean(v[h // 4:3 * h // 4, w // 4:3 * w // 4])
        edg = np.mean(np.concatenate([
            v[:h // 4].flatten(), v[3 * h // 4:].flatten(),
            v[:, :w // 4].flatten(), v[:, 3 * w // 4:].flatten()]))
        if ctr - edg > 30:
            score += min(ctr - edg - 30, 30)

        # ── CDF linearity test ──
        # Natural photos have smooth CDF; curves/levels create kinks
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
        cdf = np.cumsum(v_hist).astype(np.float64)
        cdf /= (cdf[-1] + 1e-8)
        # Fit linear between 5th and 95th percentile
        p5 = np.searchsorted(cdf, 0.05)
        p95 = np.searchsorted(cdf, 0.95)
        if p95 - p5 > 20:
            x = np.arange(p5, p95)
            y = cdf[p5:p95]
            linear = np.linspace(y[0], y[-1], len(y))
            deviation = np.mean(np.abs(y - linear))
            if deviation > 0.04:
                score += min((deviation - 0.04) * 300, 25)

        return float(np.clip(score, 0, 100))

    @staticmethod
    def run_color_coherence(image_path):
        """
        Detects color grading / filter via LAB color space analysis,
        hue distribution, and color temperature estimation.
        """
        img = cv2.imread(image_path)
        if img is None:
            return 0
        score = 0.0

        # ── LAB analysis ──
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0].astype(np.float64)
        a_ch = lab[:, :, 1].astype(np.float64)
        b_ch = lab[:, :, 2].astype(np.float64)

        # Color cast detection: natural photos have a/b centered near 128
        a_mean = np.mean(a_ch)
        b_mean = np.mean(b_ch)
        cast_dist = np.sqrt((a_mean - 128) ** 2 + (b_mean - 128) ** 2)
        if cast_dist > 12:
            score += min((cast_dist - 12) * 2.0, 25)

        # a/b channel correlation with L (color grade often ties color to tone)
        step = max(1, img.shape[0] * img.shape[1] // 30000)
        l_s = l_ch.flatten()[::step]
        a_s = a_ch.flatten()[::step]
        b_s = b_ch.flatten()[::step]

        if np.std(l_s) > 1 and np.std(a_s) > 0.5:
            corr_la = abs(np.corrcoef(l_s, a_s)[0, 1])
            # Strong L-a correlation = split toning / color grading
            if corr_la > 0.35:
                score += min((corr_la - 0.35) * 40, 20)

        if np.std(l_s) > 1 and np.std(b_s) > 0.5:
            corr_lb = abs(np.corrcoef(l_s, b_s)[0, 1])
            if corr_lb > 0.35:
                score += min((corr_lb - 0.35) * 40, 20)

        # ── Hue distribution analysis ──
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(np.float64)
        s_ch = hsv[:, :, 1].astype(np.float64)

        # Only consider sufficiently saturated pixels for hue analysis
        sat_mask = s_ch > 30
        if np.sum(sat_mask) > 500:
            hues = h_ch[sat_mask]
            h_hist = np.histogram(hues, bins=36, range=(0, 180))[0]
            h_hist_n = h_hist / (h_hist.sum() + 1e-8)
            # Hue entropy: color grading often concentrates hues
            h_entropy = -np.sum(h_hist_n[h_hist_n > 0] * np.log2(h_hist_n[h_hist_n > 0]))
            max_entropy = np.log2(36)
            # Very low entropy = heavily color graded
            if h_entropy < max_entropy * 0.45:
                score += min((max_entropy * 0.45 - h_entropy) * 8, 20)

        # ── Color temperature: warm/cool bias ──
        b_mean_rgb = np.mean(img[:, :, 0].astype(np.float64))
        r_mean_rgb = np.mean(img[:, :, 2].astype(np.float64))
        temp_bias = abs(r_mean_rgb - b_mean_rgb)
        if temp_bias > 25:
            score += min((temp_bias - 25) * 0.5, 15)

        return float(np.clip(score, 0, 100))

    # ═══════════════════════════════════════════
    #  EDIT DETECTION — BLOCK NOISE INCONSISTENCY
    # ═══════════════════════════════════════════

    @staticmethod
    def run_noise_inconsistency(image_path):
        """
        Divides image into blocks and measures noise per block.
        Returns (score, noise_heatmap_512x512).
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, None

        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (512, 512))
        blurred = cv2.medianBlur(img, 5)
        noise = cv2.absdiff(img, blurred).astype(np.float64)

        bs = 32
        block_stds = []
        noise_map = np.zeros((512, 512), dtype=np.float64)
        for i in range(0, 512 - bs, bs):
            for j in range(0, 512 - bs, bs):
                s = np.std(noise[i:i + bs, j:j + bs])
                block_stds.append(s)
                noise_map[i:i + bs, j:j + bs] = s

        block_stds = np.array(block_stds)
        cv_noise = np.std(block_stds) / (np.mean(block_stds) + 1e-8)

        raw = (cv_noise - 0.8) * 60
        score = float(np.clip(raw, 0, 100))

        # Normalize noise_map to 0-255 for visualization
        noise_map = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        noise_map = cv2.resize(noise_map, (w_orig, h_orig))

        return score, noise_map

    # ═══════════════════════════════════════════
    #  EDIT DETECTION — SHARPENING / UNSHARP MASK
    # ═══════════════════════════════════════════

    @staticmethod
    def run_sharpening_detection(image_path):
        """
        Detects artificial sharpening (Unsharp Mask, High Pass).
        Sharpened images have bright halos around edges.
        We detect this by comparing Laplacian energy to gradient energy.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0

        img = cv2.resize(img, (512, 512)).astype(np.float64)

        # Laplacian (2nd derivative — amplified by sharpening)
        lap = np.abs(cv2.Laplacian(img, cv2.CV_64F))
        # Gradient (1st derivative — less affected by sharpening)
        gx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        gy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        grad = (gx + gy) / 2

        # Ratio of Laplacian to gradient energy
        # Natural: ratio ≈ 0.3-0.8
        # Sharpened: ratio > 1.0 (Laplacian amplified by sharpening halos)
        ratio = np.mean(lap) / (np.mean(grad) + 1e-8)

        raw = (ratio - 0.8) * 100
        return float(np.clip(raw, 0, 100))

    # ═══════════════════════════════════════════
    #  EDIT DETECTION — COPY-MOVE (simplified)
    # ═══════════════════════════════════════════

    @staticmethod
    def run_copy_move_detection(image_path):
        """
        Simplified copy-move detection using block matching.
        Divides image into overlapping blocks, computes DCT features,
        and checks for highly similar non-adjacent blocks.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0

        img = cv2.resize(img, (256, 256)).astype(np.float64)
        bs = 16
        step = 8
        features = []
        positions = []

        for i in range(0, 256 - bs, step):
            for j in range(0, 256 - bs, step):
                block = img[i:i + bs, j:j + bs]
                # Use DCT coefficients as compact feature
                dct = cv2.dct(block)
                # Take top-left 4x4 low-frequency coefficients
                feat = dct[:4, :4].flatten()
                features.append(feat)
                positions.append((i, j))

        features = np.array(features)
        n = len(features)
        if n < 10:
            return 0

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        # Sort by first few DCT coefficients for efficient matching
        sort_idx = np.lexsort(features[:, :3].T)
        features = features[sort_idx]
        positions = [positions[i] for i in sort_idx]

        # Check adjacent entries in sorted list for matches
        matches = 0
        for i in range(n - 1):
            sim = np.dot(features[i], features[i + 1])
            if sim > 0.998:  # Extremely similar blocks only
                dy = abs(positions[i][0] - positions[i + 1][0])
                dx = abs(positions[i][1] - positions[i + 1][1])
                if dy > bs * 3 or dx > bs * 3:  # Must be far apart
                    matches += 1

        # Need at least 5 distant matches to flag
        if matches < 5:
            return 0
        raw = min((matches - 5) * 8, 100)
        return float(raw)

    # ═══════════════════════════════════════════
    #  AI DETECTION — EXIF + FORMAT + RESOLUTION
    # ═══════════════════════════════════════════

    @staticmethod
    def run_exif_analysis(image_path):
        """
        Checks EXIF for camera data + file format + resolution patterns.
        Returns: (ai_score, edit_software_detected)
        """
        ext = os.path.splitext(image_path)[1].lower()

        try:
            img = Image.open(image_path)
            exif_raw = img._getexif()
            img_w, img_h = img.size
        except Exception:
            exif_raw = None
            img_w, img_h = 0, 0

        # ── Resolution check ──
        ai_resolutions = [
            (1024, 1024), (1792, 1024), (1024, 1792),
            (512, 512), (768, 768), (2048, 2048),
            (1536, 1024), (1024, 1536), (1344, 768), (768, 1344),
        ]
        res_match = any(img_w == aw and img_h == ah for aw, ah in ai_resolutions)
        res_bonus = 15 if res_match else 0

        if exif_raw is None:
            # No EXIF at all
            if ext in ('.png', '.webp'):
                return 85 + res_bonus, False   # PNG/WebP + no EXIF = very likely AI
            return 70 + res_bonus, False       # JPEG + no EXIF = suspicious

        exif = {}
        for tag_id, val in exif_raw.items():
            exif[TAGS.get(tag_id, tag_id)] = val

        camera_fields = ['Make', 'Model', 'LensModel', 'LensMake']
        shooting_fields = ['ExposureTime', 'FNumber', 'ISOSpeedRatings',
                           'FocalLength', 'FocalLengthIn35mmFilm']

        cam = sum(1 for f in camera_fields if f in exif)
        shoot = sum(1 for f in shooting_fields if f in exif)
        gps = 1 if 'GPSInfo' in exif else 0
        total = cam + shoot + gps

        if total >= 5:
            ai_score = 0
        elif total >= 3:
            ai_score = 5
        elif total >= 1:
            ai_score = 25
        else:
            ai_score = 55 + res_bonus

        # Edit software detection
        software = exif.get('Software', '')
        edit_sw = False
        if isinstance(software, str):
            kws = ['photoshop', 'lightroom', 'gimp', 'snapseed', 'vsco',
                   'afterlight', 'pixlr', 'canva', 'adobe', 'capture one']
            edit_sw = any(k in software.lower() for k in kws)

        return min(ai_score, 100), edit_sw

    # ═══════════════════════════════════════════
    #  AI DETECTION — DEEP FEATURES (confirmatory)
    # ═══════════════════════════════════════════

    @classmethod
    def _load_resnet(cls):
        if cls._resnet is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            cls._resnet = nn.Sequential(*list(base.children())[:-1]).to(cls._device)
            cls._resnet.eval()
        return cls._resnet, cls._device

    @classmethod
    def run_deep_features(cls, image_path):
        """
        Confirmatory signal only — capped at 25%.
        Analyses feature activation patterns from ResNet-50 penultimate layer.
        """
        try:
            model, device = cls._load_resnet()
            tf = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img = Image.open(image_path).convert('RGB')
            t = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model(t).squeeze().cpu().numpy()

            # Activation density: fraction of features that are "active" (> 0.5)
            active = np.mean(feats > 0.5)
            # AI images tend to activate fewer features more strongly
            # Natural photos: active ≈ 0.15-0.35
            # AI: active often < 0.12 or > 0.40
            if active < 0.10 or active > 0.45:
                return 25
            elif active < 0.13 or active > 0.40:
                return 15
            return 5
        except Exception:
            return 0

    # ═══════════════════════════════════════════
    #  AI DETECTION — JPEG GHOST
    # ═══════════════════════════════════════════

    @staticmethod
    def run_jpeg_ghost(image_path):
        """Detects double JPEG compression or non-JPEG source."""
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ('.png', '.webp', '.bmp', '.tif', '.tiff'):
            # Non-JPEG source — slightly suspicious for AI
            return 15

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception:
            return 0

        img_arr = np.array(img, dtype=np.float64)
        errors = []
        for q in range(60, 100, 5):
            tmp = os.path.join(os.path.dirname(image_path) or '.', '_tmp_jg.jpg')
            img.save(tmp, 'JPEG', quality=q)
            re = np.array(Image.open(tmp), dtype=np.float64)
            if os.path.exists(tmp):
                os.remove(tmp)
            errors.append(np.mean(np.abs(img_arr - re)))

        errors = np.array(errors)
        rng = np.max(errors) - np.min(errors)

        if rng < 0.3:
            return 10  # Flat curve = likely PNG origin or very high quality
        min_q = 60 + np.argmin(errors) * 5
        if min_q < 75:
            return min(30 + (75 - min_q), 60)
        return 5

    # ═══════════════════════════════════════════
    #  DEEPFAKE — FACE ANALYSIS
    # ═══════════════════════════════════════════

    @staticmethod
    def detect_deepfake(image_path):
        """
        Returns (score, face_data_list).
        face_data_list: [(x, y, w, h, lap_var, face_score), ...]
        """
        img = cv2.imread(image_path)
        if img is None:
            return 0, []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return 0, []

        scores = []
        face_data = []
        for (fx, fy, fw, fh) in faces:
            lap = cv2.Laplacian(gray[fy:fy+fh, fx:fx+fw], cv2.CV_64F).var()
            if lap < 15:
                s = 85
            elif lap < 30:
                s = 35
            else:
                s = 0
            scores.append(s)
            face_data.append((int(fx), int(fy), int(fw), int(fh), float(lap), s))

        return float(np.mean(scores)), face_data
