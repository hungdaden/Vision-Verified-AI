import os
import subprocess
import shutil
import time
import sys

def create_desktop_shortcut(exe_path):
    """Tạo shortcut ra màn hình Desktop (chỉ chạy trên Windows)."""
    try:
        import pythoncom
        from win32com.client import Dispatch
        
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        path = os.path.join(desktop, "Vision Verify AI.lnk")
        target = exe_path
        wDir = os.path.dirname(exe_path)
        icon = exe_path # Sử dụng icon tích hợp trong exe

        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        print(f"Created desktop shortcut: {path}")
    except Exception as e:
        print(f"Could not create desktop shortcut: {e}")

def clean_folder(folder_path):
    """Xóa thư mục một cách an toàn, thử lại nếu bị khóa."""
    if os.path.exists(folder_path):
        print(f"Cleaning {folder_path}...")
        for i in range(5):  # Thử lại 5 lần nếu bị khóa
            try:
                shutil.rmtree(folder_path)
                break
            except Exception as e:
                if i == 4:
                    print(f"Warning: Could not fully clean {folder_path}: {e}")
                else:
                    time.sleep(1)

def build():
    print("=== Starting Build Process (Vision Verify AI) ===")
    
    # 1. Dọn dẹp bản build cũ triệt để
    folders_to_clean = ['build', 'dist']
    
    for folder in folders_to_clean:
        clean_folder(folder)
        
    # Tạo lại thư mục dist để đảm bảo sạch sẽ
    if not os.path.exists('dist'):
        os.makedirs('dist')

    # 2. Chạy PyInstaller
    print("Running PyInstaller...")
    try:
        result = subprocess.run(
            ['pyinstaller', '--noconfirm', '--clean', 'vision_verify.spec'], 
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        exe_path = os.path.abspath('dist/VisionVerifyAI.exe')
        print("\n=== Build Successful! ===")
        print(f"Location: {exe_path}")
        
        # 3. Tạo shortcut ra Desktop
        create_desktop_shortcut(exe_path)
        
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Build FAILED !!!")
        print(f"Error output:\n{e.stderr}")
        print("\nTip: Nếu lỗi 'Invalid argument', hãy thử tắt tạm thời Windows Real-time Protection hoặc xóa thủ công thư mục 'build' và chạy lại.")

if __name__ == "__main__":
    build()
