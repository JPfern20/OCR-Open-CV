#!/usr/bin/env python3
"""
EmTechScan – Classical OCR (Template + ML Hybrid)
-------------------------------------------------
- Uses Template Matching or HOG + kNN for OCR (No Deep Learning)
- Supports image and PDF input
- Saves recognized text to Word (.docx) or Text (.txt)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
from docx import Document
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from pdf2image import convert_from_path


class TemplateOCR:
    def __init__(self, train_dir=None):
        self.templates = {}
        self.train_dir = train_dir
        self.model = None
        self.use_ml = False  # False = template mode, True = ML mode

    # --- Dataset Loading ---
    def train(self):
        """Load all template images and labels."""
        self.templates = {}
        if not self.train_dir or not os.path.isdir(self.train_dir):
            raise ValueError("Invalid training directory")

        for file in os.listdir(self.train_dir):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            label = os.path.splitext(file)[0].split('_')[0]
            path = os.path.join(self.train_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img = cv2.resize(img, (64, 64))
            self.templates.setdefault(label, []).append(img)

        total = sum(len(v) for v in self.templates.values())
        return total

    # --- Preprocessing ---
    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file")

        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Deskew
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = thresh.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        return thresh

    # --- ML Training (HOG + kNN) ---
    def train_ml(self, n_neighbors=4):
        X, y = [], []
        for label, tmpl_list in self.templates.items():
            for img in tmpl_list:
                feat = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9)
                X.append(feat)
                y.append(label)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto')
        self.model.fit(X, y)
        self.use_ml = True
        return len(X)

    # --- OCR Recognition ---
    def recognize(self, img_path, conf_threshold=0.7):
        if not self.templates:
            raise ValueError("Templates not trained yet")

        thresh = self.preprocess(img_path)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        recognized = ""
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            roi = cv2.resize(thresh[y:y+h, x:x+w], (64, 64))

            # Try Template Matching
            best_label, best_score = None, -1
            for label, tmpl_list in self.templates.items():
                for tmpl in tmpl_list:
                    score = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(score)
                    if max_val > best_score:
                        best_score = max_val
                        best_label = label

        # If weak match and ML model available → fallback
            if best_score < conf_threshold and self.model:
                feat = hog(roi, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9)
                ml_label = self.model.predict([feat])[0]
                recognized += ml_label
            else:
                recognized += best_label if best_label else "?"

        return recognized



class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmTechScan – Classical OCR Engine")
        self.ocr = TemplateOCR()
        self.image_path = None
        self.tk_img = None
        self.result_text = ""
        self.setup_ui()

    # --- UI ---
    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="EmTechScan OCR (Template / ML Hybrid)", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=8)
        self.canvas = tk.Canvas(frm, width=480, height=360, bg="gray20")
        self.canvas.grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Button(frm, text="Select Training Folder", command=self.select_training).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(frm, text="Select Image/PDF", command=self.select_image).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Run OCR", command=self.run_ocr).grid(row=2, column=2, sticky="ew", pady=4)
        ttk.Button(frm, text="Train ML Model", command=self.train_ml).grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(frm, text="Recognition Mode:").grid(row=5, column=0, sticky="w", pady=3)
        self.mode_var = tk.StringVar(value="Hybrid")
        mode_menu = ttk.Combobox(frm, textvariable=self.mode_var,
                         values=["Template", "ML", "Hybrid"], state="readonly")
        mode_menu.grid(row=5, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Button(frm, text="Save Output", command=self.save_output).grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        self.status = ttk.Label(frm, text="Status: Ready")
        self.status.grid(row=6, column=0, columnspan=3, sticky="w", pady=3)

        self.text_box = tk.Text(frm, wrap="word", width=70, height=12)
        self.text_box.grid(row=7, column=0, columnspan=3, pady=5)

        frm.columnconfigure((0, 1, 2), weight=1)

    def select_training(self):
        path = filedialog.askdirectory(title="Select Training Folder")
        if not path:
            return
        self.ocr.train_dir = path
        try:
            n = self.ocr.train()
            self.status.config(text=f"Trained on {n} templates. Ready for OCR.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Image or PDF",
            filetypes=[("Supported files", "*.png *.jpg *.jpeg *.bmp *.pdf")]
        )
        if not path:
            return
        self.image_path = path
        if path.lower().endswith(".pdf"):
            pages = convert_from_path(path)
            temp = "page_temp.png"
            pages[0].save(temp, "PNG")
            path = temp
        self.show_preview(path)
        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((480, 360))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def train_ml(self):
        try:
            count = self.ocr.train_ml()
            self.status.config(text=f"Trained ML model with {count} samples (kNN).")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_ocr(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please select an image or PDF first.")
            return

        mode = self.mode_var.get()
        try:
            self.status.config(text=f"Running OCR ({mode})...")
            self.root.update()

            # --- Set mode flags ---
            if mode == "Template":
                self.ocr.use_ml = False
            elif mode == "ML":
                self.ocr.use_ml = True
            elif mode == "Hybrid":
                self.ocr.use_ml = False  # start with template mode
                result = self.ocr.recognize(self.image_path, conf_threshold=0.6)

                # If too many uncertain symbols, retry with ML mode
                if result.count("?") > len(result) * 0.3 and self.ocr.model:
                    self.ocr.use_ml = True
                    result = self.ocr.recognize(self.image_path)

                self.result_text = result
                self.text_box.delete("1.0", tk.END)
                self.text_box.insert("1.0", result)
                self.status.config(text=f"OCR complete ({mode}). {len(result)} characters recognized.")
                return

                # --- For Template or ML only ---
            result = self.ocr.recognize(self.image_path)
            self.result_text = result
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", result)
            self.status.config(text=f"OCR complete ({mode}). {len(result)} characters recognized.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def save_output(self):
        """Save recognized text as .docx or .txt."""
        if not self.result_text.strip():
            messagebox.showwarning("No text", "Please run OCR before saving.")
            return

        filetypes = [("Word Document", "*.docx"), ("Text File", "*.txt")]
        save_path = filedialog.asksaveasfilename(
            title="Save OCR Output",
            defaultextension=".docx",
            filetypes=filetypes
        )
        if not save_path:
            return

        try:
            if save_path.endswith(".docx"):
                doc = Document()
                doc.add_paragraph(self.result_text)
                doc.save(save_path)
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.result_text)

            self.status.config(text=f"Saved output: {os.path.basename(save_path)}")
            messagebox.showinfo("Saved", f"OCR output saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

def main():
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
