# library.py
# Berisi fungsi-fungsi dasar pengolahan citra secara manual
# Digunakan oleh semua notebook eksperimen

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# KONSTANTA
# ============================================================
DATASET_PATH = 'dataset'
CLASSES = ['jahe', 'kunyit', 'lengkuas']
IMG_SIZE = (128, 128)


# ============================================================
# FUNGSI MANUAL — implementasi dari nol menggunakan numpy
# ============================================================

def manual_resize(img, target_h, target_w):
    """
    Resize gambar menggunakan nearest neighbor interpolation.
    Tidak menggunakan cv2.resize().
    """
    src_h, src_w = img.shape[:2]
    row_idx = np.clip((np.arange(target_h) * src_h / target_h).astype(int), 0, src_h - 1)
    col_idx = np.clip((np.arange(target_w) * src_w / target_w).astype(int), 0, src_w - 1)
    if len(img.shape) == 3:
        return img[np.ix_(row_idx, col_idx, np.arange(img.shape[2]))]
    return img[np.ix_(row_idx, col_idx)]


def manual_bgr_to_grayscale(img_bgr):
    """
    Konversi BGR ke grayscale menggunakan rumus luminance.
    Tidak menggunakan cv2.cvtColor().
    Formula: Y = 0.114*B + 0.587*G + 0.299*R
    """
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    return (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)


def manual_gaussian_kernel(size, sigma=1.0):
    """Membuat kernel Gaussian secara manual."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def manual_convolution(img, kernel):
    """
    Konvolusi 2D manual menggunakan numpy.
    Tidak menggunakan cv2.filter2D().
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


def manual_gaussian_blur(img, kernel_size=5, sigma=1.0):
    """
    Gaussian blur manual menggunakan konvolusi.
    Tidak menggunakan cv2.GaussianBlur().
    """
    return manual_convolution(img, manual_gaussian_kernel(kernel_size, sigma))


def manual_median_blur(img, kernel_size=5):
    """
    Median blur manual.
    Tidak menggunakan cv2.medianBlur().
    """
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.median(padded[i:i+kernel_size, j:j+kernel_size])
    return output.astype(np.uint8)


def manual_histogram_equalization(img):
    """
    Histogram equalization manual.
    Tidak menggunakan cv2.equalizeHist().
    """
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float32)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255).astype(np.uint8)
    return lut[img]


def manual_clahe(img, clip_limit=2.0, tile_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) manual.
    Tidak menggunakan cv2.createCLAHE().
    """
    h, w = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    tiles_y, tiles_x = ph // tile_size, pw // tile_size
    luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.uint8)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = padded[ty*tile_size:(ty+1)*tile_size, tx*tile_size:(tx+1)*tile_size]
            hist = np.bincount(tile.flatten(), minlength=256).astype(np.float32)
            clip_val = clip_limit * tile.size / 256
            excess = np.sum(np.maximum(hist - clip_val, 0))
            hist = np.minimum(hist, clip_val) + excess / 256
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0].min()
            lut = np.round((cdf - cdf_min) / (tile.size - cdf_min) * 255)
            luts[ty, tx] = np.clip(lut, 0, 255).astype(np.uint8)

    output = np.zeros_like(padded, dtype=np.float32)
    for y in range(ph):
        for x in range(pw):
            ty = min(y // tile_size, tiles_y - 1)
            tx = min(x // tile_size, tiles_x - 1)
            output[y, x] = luts[ty, tx, padded[y, x]]
    return output[:h, :w].astype(np.uint8)


def manual_erosion(img, kernel_size=3):
    """
    Morfologi erosion manual.
    Tidak menggunakan cv2.erode().
    """
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=255)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.min(padded[i:i+kernel_size, j:j+kernel_size])
    return output


def manual_dilation(img, kernel_size=3):
    """
    Morfologi dilation manual.
    Tidak menggunakan cv2.dilate().
    """
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.max(padded[i:i+kernel_size, j:j+kernel_size])
    return output


def manual_opening(img, kernel_size=3):
    """Morfologi opening = erosion lalu dilation."""
    return manual_dilation(manual_erosion(img, kernel_size), kernel_size)


def manual_closing(img, kernel_size=3):
    """Morfologi closing = dilation lalu erosion."""
    return manual_erosion(manual_dilation(img, kernel_size), kernel_size)


# ============================================================
# FUNGSI UTILS — load, GLCM, evaluasi
# ============================================================

def load_images(dataset_path=DATASET_PATH, classes=CLASSES, img_size=IMG_SIZE):
    images, labels = [], []
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if not os.path.exists(folder):
            print(f'[WARNING] Folder tidak ditemukan: {folder}')
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(cls)
    print(f'Total gambar loaded: {len(images)}')
    return images, labels


def extract_glcm_features(img_gray):
    """
    Ekstraksi fitur GLCM multi-angle, multi-distance.
    Output: 24 fitur (6 properti x 4 statistik)
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(img_gray,
                        distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'ASM']:
        v = graycoprops(glcm, prop)
        features.extend([v.mean(), v.max(), v.min(), v.std()])
    return features

def extract_color_features(img_bgr):
    """
    Ekstraksi fitur warna berbasis HSV.
    Output: 6 fitur (mean & std untuk H, S, V)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        features.append(hsv[:, :, channel].mean())
        features.append(hsv[:, :, channel].std())
    return features

from skimage.feature import local_binary_pattern

def extract_lbp_features(img_gray, P=8, R=1):
    """
    Ekstraksi fitur LBP (Local Binary Pattern).
    Output: histogram LBP yang dinormalisasi (10 fitur untuk uniform LBP)
    """
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.tolist()

def evaluate_and_save(model, X_test, y_test, le, exp_name, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{exp_name} | {model_name} → {acc*100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title(f'{exp_name} - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs('output/cm', exist_ok=True)
    plt.savefig(f'output/cm/{exp_name}_{model_name}.png')
    plt.show()
    plt.close()

    return {'Eksperimen': exp_name, 'Model': model_name, 'Akurasi': round(acc * 100, 2)}


def preview_preprocessing(images, labels, preprocess_fn, exp_name, n=3):
    """Tampilkan preview original vs preprocessed untuk n gambar per kelas."""
    shown = {}
    samples = []
    for img, label in zip(images, labels):
        if label not in shown:
            shown[label] = True
            samples.append((img, label))
        if len(samples) == n:
            break

    fig, axes = plt.subplots(2, n, figsize=(4*n, 6))
    for idx, (img, label) in enumerate(samples):
        axes[0][idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0][idx].set_title(f'Original\n{label}')
        axes[0][idx].axis('off')
        axes[1][idx].imshow(preprocess_fn(img.copy()), cmap='gray')
        axes[1][idx].set_title(f'Preprocessed\n{label}')
        axes[1][idx].axis('off')

    plt.suptitle(f'Preview Preprocessing — {exp_name}')
    plt.tight_layout()
    plt.show()
