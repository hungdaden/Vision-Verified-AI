from PIL import Image
import os

def convert_to_ico():
    input_path = "assets/icon.png"
    output_path = "assets/icon.ico"
    
    if os.path.exists(input_path):
        img = Image.open(input_path)
        # Tạo icon với nhiều kích cỡ khác nhau cho Windows
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(output_path, sizes=icon_sizes)
        print(f"Icon converted and saved to {output_path}")
    else:
        print("Error: assets/icon.png not found")

if __name__ == "__main__":
    convert_to_ico()
