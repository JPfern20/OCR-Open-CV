import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# === Preprocessing ===
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (20, 20))
    return resized

# === Feature Extraction ===
def extract_features(img):
    features = hog(img, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), visualize=False)
    return features

# === Dataset Loading ===
def load_dataset(dataset_path='dataset'):
    X, y = [], []
    for font_folder in os.listdir(dataset_path):
        font_path = os.path.join(dataset_path, font_folder)
        for label in os.listdir(font_path):
            label_path = os.path.join(font_path, label)
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                try:
                    img = preprocess_image(img_path)
                    features = extract_features(img)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
    return np.array(X), np.array(y)

# === Model Training ===
def train_model(X, y, model_path='ocr_model.pkl', scaler_path='scaler.pkl'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear', probability=False)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    accuracy = clf.score(X_test, y_test)
    print(f"EmTech_Scan: Model trained...successfull! Accuracy: {accuracy:.2f}")
    return clf

# === Prediction ===
def predict_character(img_path, model_path='ocr_model.pkl', scaler_path='scaler.pkl'):
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    img = preprocess_image(img_path)
    features = extract_features(img)
    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)
    return decode_label(prediction[0])

# === Label Decoding ===
def decode_label(label):
    return chr(int(label.split("_")[1])) if label.startswith("char_") else label

# === Main Execution ===
if __name__ == "__main__":
    X, y = load_dataset()
    train_model(X, y)
