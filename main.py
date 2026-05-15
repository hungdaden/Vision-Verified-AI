import sys
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal

from ui_main import MainWindow
from engine.forensics import ForensicEngine
from engine.fusion import FusionEngine, generate_heatmap


class AnalysisWorker(QThread):
    finished = pyqtSignal(dict, object, object)
    status = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            p = self.image_path
            s = {}

            self.status.emit("ELA Analysis...")
            ela_img, s['ela'] = ForensicEngine.run_ela(p)

            self.status.emit("Multi-scale ELA...")
            s['multiscale_ela'] = ForensicEngine.run_multiscale_ela(p)

            self.status.emit("Noise Analysis...")
            _, s['noise'] = ForensicEngine.run_noise_analysis(p)

            self.status.emit("Block Noise Inconsistency Scan...")
            s['noise_inconsistency'], noise_map = ForensicEngine.run_noise_inconsistency(p)

            self.status.emit("Sharpening / Unsharp Mask Detection...")
            s['sharpening'] = ForensicEngine.run_sharpening_detection(p)

            self.status.emit("EXIF + Format + Resolution Check...")
            exif_ai, edit_sw = ForensicEngine.run_exif_analysis(p)
            s['exif_ai'] = exif_ai
            s['edit_software'] = 100 if edit_sw else 0

            self.status.emit("Deep Feature Analysis (ResNet-50)...")
            s['deep'] = ForensicEngine.run_deep_features(p)

            self.status.emit("JPEG Ghost Detection...")
            s['jpeg_ghost'] = ForensicEngine.run_jpeg_ghost(p)

            self.status.emit("Color Distribution Analysis...")
            s['color_dist'] = ForensicEngine.run_color_distribution(p)

            self.status.emit("Saturation & Contrast Analysis...")
            s['sat_contrast'] = ForensicEngine.run_saturation_contrast(p)

            self.status.emit("Color Coherence Analysis...")
            s['color_coherence'] = ForensicEngine.run_color_coherence(p)

            self.status.emit("Deepfake Analysis...")
            s['deepfake'], face_data = ForensicEngine.detect_deepfake(p)

            print("\n=== Forensic Scores ===")
            for k, v in s.items():
                print(f"  {k:20s}: {v:6.1f}")

            self.status.emit("Fusing Results...")
            results = FusionEngine.calculate_results(s)

            print(f"\n=== Final Verdict ===")
            for k, v in results.items():
                print(f"  {k:20s}: {v:6.1f}%")
            print()

            self.status.emit("Rendering Composite Heatmap...")
            heatmap = generate_heatmap(p, ela_img, noise_map, face_data, results)

            self.finished.emit(results, heatmap, ela_img)
            self.status.emit(
                f"Done — AI: {results['ai_generated']:.0f}%  "
                f"Edited: {results['photoshop_edited']:.0f}%  "
                f"Deepfake: {results['deepfake']:.0f}%")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.emit(f"Error: {str(e)}")


class VisionVerifyApp(MainWindow):
    def __init__(self):
        super().__init__()
        self.upload_btn.clicked.connect(self.open_file)
        self.view_btn.clicked.connect(self.toggle_view)
        self.view_btn.setEnabled(False)  # Disabled until analysis completes
        self.current_image_path = None
        self.original_pixmap = None
        self.heatmap_pixmap = None
        self.is_heatmap_visible = False

    def open_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if fp:
            self.process_image(fp)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls()]
        if files:
            self.process_image(files[0])

    def process_image(self, fp):
        self.current_image_path = fp
        self.original_pixmap = QPixmap(fp)
        self.set_display_image(self.original_pixmap)
        self.is_heatmap_visible = False
        self.heatmap_pixmap = None
        self.view_btn.setEnabled(False)   # Disable until new analysis finishes
        self.view_btn.setText("Show Heatmap")
        self.worker = AnalysisWorker(fp)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def update_status(self, msg):
        self.status_label.setText(msg)

    def on_analysis_finished(self, results, heatmap, ela_img):
        self.update_results(results)
        self._last_heatmap_data = heatmap.copy()
        h, w, _ = self._last_heatmap_data.shape
        q_img = QImage(self._last_heatmap_data.data, w, h, 3 * w,
                       QImage.Format.Format_BGR888)
        self.heatmap_pixmap = QPixmap.fromImage(q_img)
        self.view_btn.setEnabled(True)    # Now safe to view heatmap

    def toggle_view(self):
        if not self.heatmap_pixmap:
            return
        if self.is_heatmap_visible:
            self.set_display_image(self.original_pixmap)
            self.view_btn.setText("Show Heatmap")
        else:
            self.set_display_image(self.heatmap_pixmap)
            self.view_btn.setText("Show Original")
        self.is_heatmap_visible = not self.is_heatmap_visible


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionVerifyApp()
    window.showMaximized()
    sys.exit(app.exec())
