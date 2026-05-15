# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Đường dẫn gốc của project
pro_root = os.path.abspath('.')

added_files = [
    ('ui_main.py', '.'),
    ('engine/*.py', 'engine'),
    ('assets/*', 'assets'),
    # Thêm các file haarcascades từ opencv-python nếu cần
]

a = Analysis(
    ['main.py'],
    pathex=[pro_root],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'torch', 
        'torchvision', 
        'cv2', 
        'numpy', 
        'PIL', 
        'PyQt6',
        'engine.forensics',
        'engine.fusion'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VisionVerifyAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Đặt thành False để không hiện terminal khi mở app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)
