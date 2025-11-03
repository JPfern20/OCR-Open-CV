#!/usr/bin/env python3
"""
Template OCR Dataset Generator (Full Charset, Augmented)
--------------------------------------------------------
Generates multiple character templates (A–Z, a–z, 0–9, symbols)
with small variations (rotation, scale, blur, noise).

Output:
    ./train_data/*.png  (e.g. A_0.png, a_2.png, sym_33.png, etc.)

Dependencies:
    pip install pillow opencv-python numpy
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def generate_dataset(font_path="./fonts/arial.ttf",
                     output_dir="./train_data",
                     charset=None, img_size=128,
                     font_size=100, samples_per_char=50):

    # Full ASCII set: letters, numbers, and printable symbols
    if charset is None:
        charset = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            "!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\`~"
        )

    os.makedirs(output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise RuntimeError(f"Font not found or invalid: {font_path}\n{e}")

    for char in charset:
        for i in range(samples_per_char):
            img = Image.new("L", (img_size, img_size), color=255)
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(char, font=font)
            draw.text(((img_size - w) / 2, (img_size - h) / 2), char, font=font, fill=0)

            # Augmentations
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, fillcolor=255)

            scale = random.uniform(0.85, 1.15)
            new_size = int(img_size * scale)
            img = img.resize((new_size, new_size), Image.LANCZOS)
            canvas = Image.new("L", (img_size, img_size), color=255)
            offset = ((img_size - new_size) // 2, (img_size - new_size) // 2)
            canvas.paste(img, offset)
            img = canvas

            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.7, 1.2)))

            if random.random() < 0.6:
                arr = np.array(img, dtype=np.uint8)
                noise = np.random.randint(0, 20, arr.shape, dtype=np.uint8)
                arr = np.clip(arr + noise, 0, 255)
                img = Image.fromarray(arr)

            # Safe filenames
            safe_char = char if char.isalnum() else f"sym_{ord(char)}"
            filename = f"{safe_char}_{i}.png"
            img.save(os.path.join(output_dir, filename))

    print(f"✅ Generated {len(charset) * samples_per_char} character images in '{output_dir}'")


if __name__ == "__main__":
    generate_dataset()
