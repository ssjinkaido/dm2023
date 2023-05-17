import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import time
import argparse
from scipy.spatial.distance import cdist


class MeanShiftSegmentation:
    def __init__(self, image_path, r, threshold):
        self.image_path = image_path
        self.image = None
        self.image_lab = None
        self.segmented_image = None
        self.peaks = []
        self.r = r
        self.threshold = threshold

    def gaussian_kernel(self, sigma, size):
        mu = np.floor([size / 2, size / 2])
        size = int(size)
        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(
                    -(0.5 / (sigma * sigma))
                    * (np.square(i - mu[0]) + np.square(j - mu[1]))
                ) / (np.sqrt(2 * math.pi) * sigma * sigma)

        kernel = kernel / np.sum(kernel)
        return kernel

    def load_and_preprocess_image(self):
        self.image = plt.imread(self.image_path)
        height, width, _ = self.image.shape
        # self.image = cv2.resize(self.image, (width // 2, height // 2))
        blur_kernel = self.gaussian_kernel(5, 5)
        self.image_blur = cv2.filter2D(self.image, -1, blur_kernel)
        self.image_lab = cv2.cvtColor(self.image_blur, cv2.COLOR_BGR2LAB)

    def mean_shift_segmentation(self):
        self.image_lab = self.image_lab.astype(np.float64)
        height, width, _ = self.image_lab.shape

        X = self.image_lab.reshape(-1, 3)

        segmented_map = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            peak, _ = self.find_peak(X, X[i, :], self.r)
            while True:
                new_peak, _ = self.find_peak(X, peak, self.r)
                if np.linalg.norm(new_peak - peak) < self.threshold:
                    checker = False
                    if self.peaks:
                        for index, p in enumerate(self.peaks):
                            if np.linalg.norm(p - new_peak) < self.r / 2:
                                segmented_map[i] = index + 1
                                checker = True
                                break
                    if not checker:
                        self.peaks.append(new_peak)
                        segmented_map[i] = len(self.peaks)
                    break
                peak = new_peak

        self.segmented_image = segmented_map.reshape(height, width)

    def mean_shift_optimize(self):
        self.image_lab = self.image_lab.astype(np.float64)
        height, width, _ = self.image_lab.shape

        X = self.image_lab.reshape(-1, 3)

        segmented_map = np.negative(np.ones(X.shape[0]))
        peaks = []

        for i in range(X.shape[0]):
            if segmented_map[i] >= 0:
                continue
            checker = False
            peak, indx = self.find_peak(X, X[i, :], self.r)
            while True:
                new_peak, new_indx = self.find_peak(X, peak, self.r)
                if np.linalg.norm(new_peak - peak) < self.threshold:
                    break
                else:
                    peak = new_peak

            for index, p in enumerate(peaks):
                if np.linalg.norm(p - new_peak) < self.r / 2:
                    peaks[index] = new_peak
                    segmented_map[new_indx] = index
                    checker = True
                    break

            if not checker:
                if len(peaks) == 0:
                    peaks.append(new_peak)
                    segmented_map[new_indx] = len(peaks)
                else:
                    peaks.append(new_peak)
                    segmented_map[new_indx] = len(peaks)

        self.segmented_image = segmented_map.reshape(height, width)

    def find_peak(self, X, X1, r):
        dist = cdist(X, X1.reshape(1, -1))
        indx = np.where(dist < r)[0]

        if len(indx) == 1:
            peak = X[indx[0], :]
        else:
            peak = np.mean(X[indx, :], axis=0)
        return peak, indx

    def visualize_segmentation(self, saved_result):
        print(f"Number of clusters: {len(np.unique(self.segmented_image))}")
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        axes[0].imshow(self.image)
        axes[0].axis("off")
        axes[0].set_title("Original image")
        axes[1].axis("off")
        axes[1].imshow(self.segmented_image)
        axes[1].set_title("Segmented image")
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(saved_result)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean Shift Image Segmentation")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--radius", type=float, help="Radius value for the algorithm")
    parser.add_argument(
        "--threshold", type=float, help="Threshold value for the algorithm"
    )
    parser.add_argument("--saved_image_path", type=str, help="Path to the saved image")
    parser.add_argument(
        "--algo_type", type=bool, help="Algorithm type (vanilla or optimized)"
    )
    args = parser.parse_args()

    start = time.time()

    meanshift = MeanShiftSegmentation(
        args.image_path, r=args.radius, threshold=args.threshold
    )
    meanshift.load_and_preprocess_image()
    if not args.algo_type:
        meanshift.mean_shift_segmentation()
    else:
        meanshift.mean_shift_optimize()

    meanshift.visualize_segmentation(args.saved_image_path)

    print(f"Running time: {(time.time()-start):.4f}")
