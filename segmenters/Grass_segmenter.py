import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image

from Segmenter import Segmenter


class GrassSegmenter(Segmenter):
    def __init__(self, class_info, lbp_radius=1, lbp_points=8, lbp_threshold=0.5):
        super().__init__(class_info)
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_threshold = lbp_threshold  # Threshold for texture-based segmentation

    def segment(self, image, mask=None):
        # Ensure image is a NumPy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert image to grayscale if it is in color
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Compute LBP for the grayscale image
        lbp_image = self._compute_lbp(gray_image)

        # Create a binary mask by thresholding the LBP values
        _, binary_mask = cv2.threshold(lbp_image, self.lbp_threshold * np.max(lbp_image), 1, cv2.THRESH_BINARY)

        # Return the binary mask (grass regions identified as 1, others as 0)
        return binary_mask.astype(np.uint8)

    def _compute_lbp(self, gray_image):
        # Compute LBP for texture analysis using uniform pattern
        lbp = local_binary_pattern(gray_image, self.lbp_points, self.lbp_radius, method='uniform')
        return lbp

    def apply_mask(self, image, mask):
        # Apply a binary mask to the image for segmentation
        return cv2.bitwise_and(image, image, mask=mask)

    def display_segmentation_result(self, processed_img_pil, processed_mask_array, result):
        """
        Display the processed image, processed mask, and result mask.
        """
        # Convert PIL Image to NumPy array
        if isinstance(processed_img_pil, Image.Image):
            processed_img = np.array(processed_img_pil)
        else:
            processed_img = processed_img_pil  # Already a NumPy array

        # Plotting images side by side
        plt.figure(figsize=(15, 5))

        # Processed Image
        plt.subplot(1, 3, 1)
        plt.title("Processed Image")
        plt.imshow(processed_img)
        plt.axis("off")

        # Processed Mask
        plt.subplot(1, 3, 2)
        plt.title("Processed Mask")
        plt.imshow(processed_mask_array, cmap="gray")
        plt.axis("off")

        # Result Mask
        plt.subplot(1, 3, 3)
        plt.title("Result Mask")
        plt.imshow(result, cmap="gray")
        plt.axis("off")

        plt.show()
