import os
import subprocess
import shutil

def build():
    print("=== Starting Build Process ===")
    
    # 1. Dọn dẹp thư mục cũ
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            print(f"Cleaning {folder}...")
            shutil.rmtree(folder)

    # 2. Chạy PyInstaller
    print("Running PyInstaller...")
    try:
        subprocess.run(['pyinstaller', '--clean', 'vision_verify.spec'], check=True)
        print("\n=== Build Successful! ===")
        print("Your app is located in: dist/VisionVerifyAI.exe")
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")

if __name__ == "__main__":
    build()
