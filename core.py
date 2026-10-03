from __future__ import annotations

import csv
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass
class ProcessConfig:
    out_width: int = 413
    out_height: int = 579
    dpi: int = 300
    face_height_ratio: float = 0.45
    face_center_y_ratio: float = 0.42
    score_threshold: float = 0.82
    auto_straighten: bool = True
    max_straighten_deg: float = 12.0
    multi_face_area_ratio: float = 0.25
    blur_threshold: float = 55.0
    flag_blur: bool = True
    jpeg_quality: int = 95
    recursive: bool = False
    enable_review: bool = True
    skip_detection_errors: bool = True


@dataclass
class FaceInfo:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    left_eye: Optional[tuple[float, float]] = None
    right_eye: Optional[tuple[float, float]] = None

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class ProcessResult:
    filename: str
    status: str
    reason: str
    detector: str
    faces: int
    blur_score: float | None = None
    output_path: str = ''


def app_base_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_base_dir() -> Path:
    if hasattr(sys, '_MEIPASS'):
        return Path(getattr(sys, '_MEIPASS'))
    return app_base_dir()


def find_yunet_model() -> Optional[Path]:
    name = 'face_detection_yunet_2023mar.onnx'
    candidates = [
        app_base_dir() / 'models' / name,
        bundled_base_dir() / 'models' / name,
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def read_image(path: Path) -> np.ndarray:
    # PIL handles EXIF orientation and Chinese/Unicode paths reliably.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_jpeg(path: Path, bgr: np.ndarray, dpi: int, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    im.save(path, 'JPEG', quality=quality, subsampling=0, dpi=(dpi, dpi), optimize=True)


def rotate_image_keep_size(img: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # BORDER_REPLICATE avoids obvious black triangles on near-uniform photo backgrounds.
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compute_crop_rect(
    img_w: int,
    img_h: int,
    face: FaceInfo,
    out_w: int,
    out_h: int,
    face_height_ratio: float,
    face_center_y_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Compute aspect-correct crop around a face.

    face_height_ratio controls how tall the detected facial box appears in the final image.
    face_center_y_ratio controls where the facial-box center lands vertically in final image.
    """
    if face.h <= 1 or img_w <= 1 or img_h <= 1:
        return None

    aspect = out_w / out_h
    crop_h = face.h / max(0.20, min(0.75, face_height_ratio))
    crop_w = crop_h * aspect

    # If crop is too large, there is not enough surrounding image to keep the requested composition.
    if crop_h > img_h * 1.005 or crop_w > img_w * 1.005:
        return None

    x0 = face.cx - crop_w / 2.0
    y0 = face.cy - crop_h * face_center_y_ratio

    # Shift the window inside source bounds while preserving composition as much as possible.
    x0 = min(max(0.0, x0), img_w - crop_w)
    y0 = min(max(0.0, y0), img_h - crop_h)

    x1 = x0 + crop_w
    y1 = y0 + crop_h

    # Round carefully, then correct any 1px overshoot.
    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = int(round(x1)), int(round(y1))
    x0i = max(0, x0i)
    y0i = max(0, y0i)
    x1i = min(img_w, x1i)
    y1i = min(img_h, y1i)
    if x1i - x0i < 10 or y1i - y0i < 10:
        return None
    return x0i, y0i, x1i, y1i


class FaceDetector:
    def __init__(self, score_threshold: float = 0.82):
        self.score_threshold = score_threshold
        self.model_path = find_yunet_model()
        self.kind = 'YuNet' if self.model_path else 'OpenCV Haar（备用）'
        self._yunet = None
        self._haar = None
        self.load_warning = ""
        if self.model_path:
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    'onnx', np.frombuffer(self.model_path.read_bytes(), dtype=np.uint8),
                    np.empty(0, dtype=np.uint8), (320, 320),
                    score_threshold, 0.3, 5000
                )
                # Confirm inference works before advertising YuNet as available.
                self._yunet.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            except Exception as exc:
                self.load_warning = f'YuNet 加载失败，已使用备用检测器：{exc}'
                self._yunet = None
                self.kind = 'OpenCV Haar（备用）'
        if self._yunet is None:
            cascade = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
            # OpenCV's filename loader cannot reliably open Chinese paths on Windows.
            # Let Python read Unicode paths and load the cascade from memory instead.
            storage = cv2.FileStorage(
                cascade.read_text(encoding='utf-8'),
                cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY,
            )
            try:
                self._haar = cv2.CascadeClassifier()
                self._haar.read(storage.getFirstTopLevelNode())
            finally:
                storage.release()
            if self._haar.empty():
                raise RuntimeError('无法加载任何人脸检测器。请安装完整 opencv-python。')

    def detect(self, bgr: np.ndarray) -> list[FaceInfo]:
        if self._yunet is not None:
            return self._detect_yunet(bgr)
        return self._detect_haar(bgr)

    def _detect_yunet(self, bgr: np.ndarray) -> list[FaceInfo]:
        h, w = bgr.shape[:2]
        # Downscale very large camera images for speed; coordinates are mapped back.
        scale = min(1.0, 1600.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        sh, sw = small.shape[:2]
        self._yunet.setInputSize((sw, sh))
        _ok, faces = self._yunet.detect(small)
        if faces is None:
            return []
        out: list[FaceInfo] = []
        inv = 1.0 / scale
        for f in faces:
            x, y, fw, fh = [float(v) * inv for v in f[:4]]
            # YuNet output landmark order: right eye, left eye, nose, right mouth, left mouth.
            right_eye = (float(f[4]) * inv, float(f[5]) * inv)
            left_eye = (float(f[6]) * inv, float(f[7]) * inv)
            score = float(f[-1])
            out.append(FaceInfo(x, y, fw, fh, score, left_eye=left_eye, right_eye=right_eye))
        return sorted(out, key=lambda z: z.area, reverse=True)

    def _detect_haar(self, bgr: np.ndarray) -> list[FaceInfo]:
        h, w = bgr.shape[:2]
        scale = min(1.0, 1400.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(48, 48))
        inv = 1.0 / scale
        out = [FaceInfo(float(x)*inv, float(y)*inv, float(fw)*inv, float(fh)*inv) for x, y, fw, fh in boxes]
        return sorted(out, key=lambda z: z.area, reverse=True)


def eye_roll_angle(face: FaceInfo) -> Optional[float]:
    if not face.left_eye or not face.right_eye:
        return None
    lx, ly = face.left_eye
    rx, ry = face.right_eye
    dx = lx - rx
    dy = ly - ry
    if abs(dx) < 1:
        return None
    # Image y-axis points downward; rotating by this angle levels the eyes.
    return math.degrees(math.atan2(dy, dx))


def face_blur_score(bgr: np.ndarray, face: FaceInfo) -> float:
    h, w = bgr.shape[:2]
    pad_x, pad_y = face.w * 0.12, face.h * 0.12
    x0 = max(0, int(face.x - pad_x))
    y0 = max(0, int(face.y - pad_y))
    x1 = min(w, int(face.x + face.w + pad_x))
    y1 = min(h, int(face.y + face.h + pad_y))
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def list_images(root: Path, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = root.rglob('*') if recursive else root.glob('*')
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


class BatchProcessor:
    def __init__(self, config: ProcessConfig):
        self.config = config
        self.detector = FaceDetector(config.score_threshold)

    def process_one(self, src: Path, out_dir: Path, review_dir: Path) -> ProcessResult:
        cfg = self.config
        try:
            img = read_image(src)
        except Exception as exc:
            self._copy_review(src, review_dir, '读取失败')
            return ProcessResult(src.name, '复核' if cfg.enable_review else '跳过', f'读取失败: {exc}', self.detector.kind, 0)

        original = img
        try:
            faces = self.detector.detect(img)
        except Exception as exc:
            return self._handle_detection_issue(src, original, out_dir, review_dir, f'人脸识别异常: {exc}', 0)
        if not faces:
            return self._handle_detection_issue(src, original, out_dir, review_dir, '未检测到人脸', 0)

        primary = faces[0]
        if len(faces) >= 2 and faces[1].area >= primary.area * cfg.multi_face_area_ratio:
            return self._handle_detection_issue(src, original, out_dir, review_dir, '检测到多个明显人脸', len(faces))

        # Optional roll correction available with YuNet landmarks.
        if cfg.auto_straighten:
            angle = eye_roll_angle(primary)
            if angle is not None and 0.6 <= abs(angle) <= cfg.max_straighten_deg:
                try:
                    img = rotate_image_keep_size(img, angle, (primary.cx, primary.cy))
                    faces2 = self.detector.detect(img)
                except Exception as exc:
                    return self._handle_detection_issue(src, original, out_dir, review_dir, f'纠斜后识别异常: {exc}', len(faces))
                if not faces2:
                    return self._handle_detection_issue(src, original, out_dir, review_dir, '纠斜后未检测到人脸', 0)
                if len(faces2) >= 2 and faces2[1].area >= faces2[0].area * cfg.multi_face_area_ratio:
                    return self._handle_detection_issue(src, original, out_dir, review_dir, '纠斜后检测到多个明显人脸', len(faces2))
                primary = faces2[0]
                faces = faces2

        blur = face_blur_score(img, primary)
        ih, iw = img.shape[:2]
        rect = compute_crop_rect(
            iw, ih, primary,
            cfg.out_width, cfg.out_height,
            cfg.face_height_ratio, cfg.face_center_y_ratio,
        )
        if rect is None:
            return self._handle_detection_issue(src, original, out_dir, review_dir, '人物过近或原图四周空间不足', len(faces), blur)

        x0, y0, x1, y1 = rect
        crop = img[y0:y1, x0:x1]
        out = cv2.resize(crop, (cfg.out_width, cfg.out_height), interpolation=cv2.INTER_LANCZOS4)
        out_path = out_dir / f'{src.stem}.jpg'
        save_jpeg(out_path, out, cfg.dpi, cfg.jpeg_quality)

        reason = '正常'
        status = '完成'
        if cfg.flag_blur and blur < cfg.blur_threshold:
            reason = f'已输出；清晰度偏低({blur:.1f})，建议复核'
            status = '完成/警告'
            self._copy_review(src, review_dir, '清晰度偏低')
        return ProcessResult(src.name, status, reason, self.detector.kind, len(faces), blur, str(out_path))

    def _handle_detection_issue(
        self, src: Path, original: np.ndarray, out_dir: Path, review_dir: Path,
        reason: str, faces: int, blur: float | None = None,
    ) -> ProcessResult:
        cfg = self.config
        self._copy_review(src, review_dir, reason)
        if cfg.skip_detection_errors:
            status = '复核' if cfg.enable_review else '跳过'
            return ProcessResult(src.name, status, f'{reason}；已跳过输出', self.detector.kind, faces, blur)

        # No reliable face composition is available: preserve proportions and
        # center-crop the original (EXIF-oriented) image to the requested size.
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        fitted = ImageOps.fit(
            Image.fromarray(rgb), (cfg.out_width, cfg.out_height),
            method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
        )
        out = cv2.cvtColor(np.asarray(fitted), cv2.COLOR_RGB2BGR)
        out_path = out_dir / f'{src.stem}.jpg'
        save_jpeg(out_path, out, cfg.dpi, cfg.jpeg_quality)
        return ProcessResult(
            src.name, '完成/警告', f'{reason}；已按原图居中裁剪直接输出（未按人脸构图）',
            self.detector.kind, faces, blur, str(out_path),
        )

    def _copy_review(self, src: Path, review_dir: Path, reason: str) -> None:
        if not self.config.enable_review:
            return
        # The report records the reason; the folder mirrors the source tree.
        dest_dir = review_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            stem, suf = dest.stem, dest.suffix
            i = 2
            while (dest_dir / f'{stem}_{i}{suf}').exists():
                i += 1
            dest = dest_dir / f'{stem}_{i}{suf}'
        try:
            shutil.copy2(src, dest)
        except Exception:
            pass

    def process_folder(
        self,
        input_dir: Path,
        output_root: Path,
        progress_cb: Optional[Callable[[int, int, ProcessResult], None]] = None,
        stop_cb: Optional[Callable[[], bool]] = None,
    ) -> tuple[list[ProcessResult], Path]:
        input_dir = input_dir.resolve()
        output_root = output_root.resolve()
        if input_dir == output_root:
            raise ValueError('输出目录不能与原始照片目录相同，请选择独立的输出文件夹。')
        # Do not process previous outputs when the output folder is inside the input.
        images = [src for src in list_images(input_dir, self.config.recursive)
                  if not src.resolve().is_relative_to(output_root)]
        out_dir = output_root / '二寸成品'
        review_dir = output_root / '人工复核'
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.config.enable_review:
            review_dir.mkdir(parents=True, exist_ok=True)

        results: list[ProcessResult] = []
        total = len(images)
        for idx, src in enumerate(images, 1):
            if stop_cb and stop_cb():
                break
            relative = src.relative_to(input_dir)
            result = self.process_one(src, out_dir / relative.parent, review_dir / relative.parent)
            result.filename = str(relative)
            results.append(result)
            if progress_cb:
                progress_cb(idx, total, result)

        report = output_root / '处理报告.csv'
        with report.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '状态', '说明', '检测器', '检测人脸数', '清晰度分数', '输出路径'])
            for r in results:
                writer.writerow([
                    r.filename, r.status, r.reason, r.detector, r.faces,
                    '' if r.blur_score is None else f'{r.blur_score:.1f}', r.output_path
                ])
        return results, report
