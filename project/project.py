import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from scipy.spatial.distance import cdist
import time

start = time.time()


def gaussian_kernel(sigma, size):
    mu = np.floor([size / 2, size / 2])
    size = int(size)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(
                -(0.5 / (sigma * sigma)) * (np.square(i - mu[0]) + np.square(j - mu[1]))
            ) / (np.sqrt(2 * math.pi) * sigma * sigma)

    kernel = kernel / np.sum(kernel)
    return kernel


def process_image():
    # Load the input image
    image = plt.imread("cow.jpg")
    height, width, _ = image.shape

    image = cv2.resize(image, (width // 2, height // 2))
    # Blur the image
    blur_kernel = gaussian_kernel(5, 5)
    image_blur = cv2.filter2D(image, -1, blur_kernel)

    # Convert to Lab color space
    image_lab = cv2.cvtColor(image_blur, cv2.COLOR_BGR2LAB)
    return image, image_lab


def mean_shift_segmentation(image):
    image = image.astype(np.float64)
    height, width, _ = image.shape

    X = image.reshape(-1, 3)

    r = 0.03
    threshold = 0.05
    segmented_map = np.zeros(X.shape[0])
    peaks = []

    for i in range(X.shape[0]):
        if i % 1000 == 0:
            print(i)
        peak, _ = find_peak(X, X[i, :], r)
        while True:
            new_peak, _ = find_peak(X, peak, r)
            if np.linalg.norm(new_peak - peak) < threshold:
                checker = False
                if peaks:
                    for index, p in enumerate(peaks):
                        if np.linalg.norm(p - new_peak) < r / 2:
                            segmented_map[i] = index + 1
                            checker = True
                            break
                if not checker:
                    peaks.append(new_peak)
                    segmented_map[i] = len(peaks)
                break
            peak = new_peak

    segmented_map = segmented_map.reshape(height, width)
    return segmented_map


def mean_shift_optimize(image):
    image = image.astype(np.float64)
    height, width, _ = image.shape

    X = image.reshape(-1, 3)

    r = 0.03
    threshold = 0.05
    segmented_map = np.negative(np.ones(X.shape[0]))
    peaks = []

    for i in range(X.shape[0]):
        if segmented_map[i] >= 0:
            continue
        checker = False
        peak, indx = find_peak(X, X[i, :], r)
        while True:
            new_peak, new_indx = find_peak(X, peak, r)
            if np.linalg.norm(new_peak - peak) < threshold:
                break
            else:
                peak = new_peak

        for index, p in enumerate(peaks):
            if np.linalg.norm(p - new_peak) < r / 2:
                peaks[index] = new_peak
                segmented_map[new_indx] = index
                checker = True
                break

        if not checker:
            peaks.append(new_peak)
            segmented_map[new_indx] = len(peaks)

    segmented_map = segmented_map.reshape(height, width)
    return segmented_map


def find_peak(X, X1, r):
    dist = cdist(X, X1.reshape(1, -1))
    indx = np.where(dist < r)[0]

    if len(indx) == 1:
        peak = X[indx[0], :]
    else:
        peak = np.mean(X[indx, :], axis=0)
    return peak, indx


image, image_lab = process_image()
segmented_image = mean_shift_segmentation(image_lab)
# segmented_image = mean_shift_optimize(image_lab)
print(f"Number of clusters: {len(np.unique(segmented_image))}")
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

axes[0].imshow(image)
axes[0].axis("off")
axes[0].set_title("Original image")
axes[1].imshow(segmented_image)
axes[1].axis("off")
axes[1].set_title("Segmented image")
plt.subplots_adjust(wspace=0.05)
plt.savefig("result.png")
plt.show()
print(f"Time running: {(time.time()-start):.4f}")
