from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QFrame, QScrollArea, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont

class ModernProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #2a2a2a;
                height: 10px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #00f2fe, stop:1 #4facfe);
            }
        """)
        self._value = 0

    def animate_to(self, value):
        self.ani = QPropertyAnimation(self, b"value")
        self.ani.setDuration(1000)
        self.ani.setStartValue(self.value())
        self.ani.setEndValue(int(value))
        self.ani.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.ani.start()

class ResultCard(QFrame):
    def __init__(self, title, percentage, color_start, color_end):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e1e1e;
                border-radius: 15px;
                border: 1px solid #333;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #aaa; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)
        
        self.percent_label = QLabel(f"{percentage}%")
        self.percent_label.setStyleSheet(f"color: white; font-size: 24px; font-weight: 900;")
        layout.addWidget(self.percent_label)
        
        self.bar = ModernProgressBar()
        self.bar.setStyleSheet(self.bar.styleSheet().replace("#00f2fe", color_start).replace("#4facfe", color_end))
        self.bar.setValue(percentage)
        layout.addWidget(self.bar)
        
        # Shadow Effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionVerify AI - Forensic Suite")
        self.setMinimumSize(900, 600)
        self.setAcceptDrops(True)
        
        # Main Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #4facfe;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00f2fe;
            }
            #DropZone {
                border: 2px dashed #444;
                border-radius: 20px;
                background-color: #1a1a1a;
            }
            #DropZone:hover {
                border-color: #4facfe;
                background-color: #222;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Header
        header = QHBoxLayout()
        title_vbox = QVBoxLayout()
        self.app_title = QLabel("VisionVerify AI")
        self.app_title.setStyleSheet("font-size: 32px; font-weight: 800; color: white;")
        self.app_subtitle = QLabel("Deep Forensic Image Analysis Engine")
        self.app_subtitle.setStyleSheet("font-size: 14px; color: #888;")
        title_vbox.addWidget(self.app_title)
        title_vbox.addWidget(self.app_subtitle)
        header.addLayout(title_vbox)
        header.addStretch()
        
        self.upload_btn = QPushButton("Import Image")
        header.addWidget(self.upload_btn)
        main_layout.addLayout(header)
        
        # Content Split (Images | Results)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left: Image Display
        self.image_container = QFrame()
        self.image_container.setObjectName("DropZone")
        self.image_container.setMinimumWidth(650)
        image_vbox = QVBoxLayout(self.image_container)
        
        self.display_label = QLabel("Drag & Drop Image Here\nor Click Import")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("font-size: 18px; color: #555;")
        image_vbox.addWidget(self.display_label)
        
        # Labels for Original vs Heatmap
        self.mode_layout = QHBoxLayout()
        self.mode_layout.setContentsMargins(10, 10, 10, 10)
        self.view_btn = QPushButton("Toggle Heatmap")
        self.view_btn.setVisible(False)
        self.mode_layout.addWidget(self.view_btn)
        image_vbox.addLayout(self.mode_layout)
        
        content_layout.addWidget(self.image_container)
        
        # Shadow for image container
        img_shadow = QGraphicsDropShadowEffect()
        img_shadow.setBlurRadius(30)
        img_shadow.setXOffset(0)
        img_shadow.setYOffset(15)
        img_shadow.setColor(QColor(0, 0, 0, 150))
        self.image_container.setGraphicsEffect(img_shadow)
        
        # Right: Dashboard
        self.dashboard = QVBoxLayout()
        self.dashboard.setSpacing(15)
        
        self.ai_card = ResultCard("AI GENERATED", 0, "#f093fb", "#f5576c")
        self.edit_card = ResultCard("PHOTOSHOP EDITED", 0, "#4facfe", "#00f2fe")
        self.fake_card = ResultCard("DEEPFAKE", 0, "#43e97b", "#38f9d7")
        
        self.dashboard.addWidget(self.ai_card)
        self.dashboard.addWidget(self.edit_card)
        self.dashboard.addWidget(self.fake_card)
        
        # Analysis Status
        self.status_box = QFrame()
        self.status_box.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; border: 1px solid #333;")
        status_layout = QVBoxLayout(self.status_box)
        self.status_label = QLabel("Ready to analyze...")
        self.status_label.setStyleSheet("color: #888; font-size: 13px;")
        status_layout.addWidget(self.status_label)
        self.dashboard.addWidget(self.status_box)
        
        self.dashboard.addStretch()
        content_layout.addLayout(self.dashboard)
        
        main_layout.addLayout(content_layout)

    def set_display_image(self, pixmap):
        self.display_label.setPixmap(pixmap.scaled(self.display_label.size(), 
                                                 Qt.AspectRatioMode.KeepAspectRatio, 
                                                 Qt.TransformationMode.SmoothTransformation))
        self.view_btn.setVisible(True)

    def update_results(self, results):
        # Update labels
        self.ai_card.percent_label.setText(f"{results['ai_generated']:.1f}%")
        self.edit_card.percent_label.setText(f"{results['photoshop_edited']:.1f}%")
        self.fake_card.percent_label.setText(f"{results['deepfake']:.1f}%")
        
        # Animate bars
        self.ai_card.bar.animate_to(results['ai_generated'])
        self.edit_card.bar.animate_to(results['photoshop_edited'])
        self.fake_card.bar.animate_to(results['deepfake'])
