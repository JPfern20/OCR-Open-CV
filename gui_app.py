import tkinter as tk
from tkinter import filedialog, Label
from ocr_engine import predict_character
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os
import cv2

import sys
import tkinter.messagebox as messagebox

# Dependency check
missing = []

try:
    import cv2
except ImportError:
    missing.append("opencv-python")

try:
    import numpy
except ImportError:
    missing.append("numpy")

try:
    import skimage
except ImportError:
    missing.append("scikit-image")

try:
    import sklearn
except ImportError:
    missing.append("scikit-learn")

try:
    import joblib
except ImportError:
    missing.append("joblib")

try:
    import PIL
except ImportError:
    missing.append("Pillow")

try:
    import reportlab
except ImportError:
    missing.append("reportlab")

if missing:
    messagebox.showwarning(
        "Missing Dependencies",
        f"The following libraries are missing:\n\n{', '.join(missing)}\n\n"
        "To make the application fully work, run:\n\npip install -r requirements.txt"
    )
    sys.exit(1)

def group_boxes_into_lines(boxes, line_threshold=15):
    lines = []
    current_line = []
    current_y = -1

    for box in sorted(boxes, key=lambda b: (b[1], b[0])):  # sort top-to-bottom, then left-to-right
        x, y, w, h = box
        if current_y == -1 or abs(y - current_y) <= line_threshold:
            current_line.append(box)
            current_y = y
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            current_y = y
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines

def predict_character_from_image(img):
    resized = cv2.resize(img, (20, 20))
    _, buf = cv2.imencode('.png', resized)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(buf)
    temp_file.close()
    try:
        char = predict_character(temp_file.name)
    except Exception:
        char = '?'
    os.remove(temp_file.name)
    return char

def scan_document(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Use connected components for stable character segmentation
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area > 10:  # filter out small noise
            boxes.append((x, y, w, h))

    grouped_lines = group_boxes_into_lines(boxes)
    lines = []

    for line_boxes in grouped_lines:
        line_text = ""
        for x, y, w, h in line_boxes:
            char_img = cleaned[y:y+h, x:x+w]
            char = predict_character_from_image(char_img)
            line_text += char
        lines.append(line_text)

    return lines

def browse_and_convert():
    file_path = filedialog.askopenfilename()
    if file_path:
        lines = scan_document(file_path)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(os.path.dirname(file_path), f"{base_name}_ocr.pdf")

        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        y = height - 50  # Start near top

        c.setFont("Times-Roman", 12)
        c.drawString(50, y, "Scanned Document Copy")
        y -= 30

        for line in lines:
            c.drawString(50, y, line)
            y -= 20
            if y < 50:
                c.showPage()
                c.setFont("Times-Roman", 12)
                y = height - 50

        c.save()
        result_label.config(text=f"PDF saved to:\n{output_path}")

# GUI Setup
root = tk.Tk()
root.title("EmTechScan")

browse_button = tk.Button(root, text="Select Image of the Document", command=browse_and_convert)
browse_button.pack(pady=10)

result_label = Label(root, text="Document OCR Result")
result_label.pack(pady=10)

root.mainloop()
