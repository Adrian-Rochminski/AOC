from Segmenter import Segmenter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


class WaterfallSegmenter(Segmenter):
    def __init__(self, class_info, num_clusters=2, window_size=5, step_size=3):
        super().__init__(class_info)
        self.num_clusters = num_clusters
        self.window_size = window_size  # Size of the patch
        self.step_size = step_size      # Step size for sliding window

    def segment(self, image):
        # Convert PIL Image to NumPy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert image to grayscale if it is in color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Estimate AR coefficients for patches
        ar_features, positions = self._compute_ar_features(gray_image)

        # Normalize the features
        ar_features = (ar_features - np.mean(ar_features, axis=0)) / (np.std(ar_features, axis=0) + 1e-5)

        # Cluster patches using KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        labels = kmeans.fit_predict(ar_features)

        # Create a segmentation mask from the labels
        segmented_image = np.zeros(gray_image.shape, dtype=np.uint8)
        count_matrix = np.zeros(gray_image.shape, dtype=np.uint8)  # To handle overlapping patches

        for idx, (i, j) in enumerate(positions):
            segmented_image[i:i+self.window_size, j:j+self.window_size] += labels[idx]
            count_matrix[i:i+self.window_size, j:j+self.window_size] += 1

        # Avoid division by zero
        count_matrix[count_matrix == 0] = 1
        segmented_image = segmented_image / count_matrix  # Average overlapping labels
        segmented_image = np.round(segmented_image).astype(np.uint8)

        # Identify the cluster corresponding to water
        height, width = segmented_image.shape
        lower_half = segmented_image[height // 2:, :]
        unique_labels, counts = np.unique(lower_half, return_counts=True)
        water_label = unique_labels[np.argmax(counts)]
        water_mask = (segmented_image == water_label).astype(np.uint8) * 255

        # Optional: Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        return water_mask

    def _compute_ar_features(self, gray_image):
        w = self.window_size
        s = self.step_size
        rows, cols = gray_image.shape

        # Pad the image to handle borders
        padded_image = np.pad(gray_image, w//2, mode='reflect')

        # Lists to hold AR coefficients and their positions
        ar_features = []
        positions = []

        # Slide window over the image
        for i in range(0, rows - w + 1, s):
            for j in range(0, cols - w + 1, s):
                # Extract patch
                patch = padded_image[i:i + w, j:j + w]

                # Prepare data for AR model
                X, y = self._prepare_ar_data(patch)

                # Estimate AR coefficients using least squares
                if X.shape[0] > 0:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    ar_features.append(coeffs.flatten())
                    positions.append((i, j))

        ar_features = np.array(ar_features)
        return ar_features, positions

    def _prepare_ar_data(self, patch):
        w = self.window_size
        # Exclude border pixels
        X = []
        y = []
        for i in range(1, w - 1):
            for j in range(1, w - 1):
                # Center pixel value
                center = patch[i, j]
                # Neighboring pixels
                neighbors = [
                    patch[i-1, j-1], patch[i-1, j], patch[i-1, j+1],
                    patch[i, j-1],               patch[i, j+1],
                    patch[i+1, j-1], patch[i+1, j], patch[i+1, j+1]
                ]
                X.append(neighbors)
                y.append(center)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def display_segmentation_result(self, original_image, result_mask, m):
        """
        Display the original image and result mask side by side.
        """
        # Convert PIL Image to NumPy array if necessary
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # Plotting images side by side
        plt.figure(figsize=(10, 5))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            plt.imshow(original_image)
        else:
            plt.imshow(original_image, cmap='gray')
        plt.axis("off")

        # Result Mask
        plt.subplot(1, 2, 2)
        plt.title("Water Segmentation Mask")
        plt.imshow(result_mask, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    def apply_mask(self, image, mask):
        pass
