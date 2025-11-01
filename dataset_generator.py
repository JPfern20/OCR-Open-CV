import os
from PIL import Image, ImageDraw, ImageFont

# Define character set
UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LOWERCASE = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
SPECIALS = '!@#$%^&*()-_+=,.?/'

CHARACTERS = UPPERCASE + LOWERCASE + DIGITS + SPECIALS

# Font folder and output
FONT_DIR = 'fonts'
OUTPUT_DIR = 'dataset'
IMG_SIZE = (100, 100)
FONT_SIZE = 72

def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for font_file in os.listdir(FONT_DIR):
        if font_file.endswith('.ttf'):
            font_path = os.path.join(FONT_DIR, font_file)
            font_name = os.path.splitext(font_file)[0]

            try:
                font = ImageFont.truetype(font_path, FONT_SIZE)
            except Exception as e:
                print(f"Failed to load font {font_file}: {e}")
                continue

            for char in CHARACTERS:
                safe_char = char if char.isalnum() else f"char_{ord(char)}"
                char_folder = os.path.join(OUTPUT_DIR, font_name, safe_char)
                os.makedirs(char_folder, exist_ok=True)

                img = Image.new('L', IMG_SIZE, color=255)
                draw = ImageDraw.Draw(img)
                w, h = draw.textsize(char, font=font)
                draw.text(((IMG_SIZE[0]-w)//2, (IMG_SIZE[1]-h)//2), char, font=font, fill=0)

                filename = f"{safe_char}_{len(os.listdir(char_folder))}.png"
                img.save(os.path.join(char_folder, filename))

    print("Emtech_Scan: DataJob is done.. Dataset is now ready. Thank you.")

if __name__ == "__main__":
    generate_dataset()
