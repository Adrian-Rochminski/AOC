import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

from Segmenter import Segmenter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.tsa.ar_model import AutoReg
import cv2


class WaterSegmenter(Segmenter):
    def __init__(self, class_info):
        super().__init__(class_info)

    def train_ar_model(self, texture_region, lags=1):
        """
        Train an autoregressive model using a given texture region of the image.

        Parameters:
        - texture_region: np.array, grayscale region of the image with desired texture.
        - lags: int, number of lags for AR model.

        Returns:
        - AR model coefficients
        """
        texture_flattened = texture_region.flatten()
        ar_model = AutoReg(texture_flattened, lags=lags).fit()
        return ar_model.params

    def segment(self, image,mask=None):
        """
        Segment water texture based on AR model coefficients.

        Parameters:
        - image: PIL Image or NumPy array

        Returns:
        - Segmentation mask
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

            # Convert to grayscale if necessary
        if len(image.shape) == 3:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = image

            # Define a sample region with water texture and train AR model
        water_texture_sample = grayscale_image[50:100, 50:100]  # Adjust based on image content
        ar_params = self.train_ar_model(water_texture_sample, lags=1)

        # Segmenting the entire image
        mask = np.zeros(grayscale_image.shape, dtype=np.uint8)
        for i in range(1, grayscale_image.shape[0] - 1):
            for j in range(1, grayscale_image.shape[1] - 1):
                # Extract a row or column neighborhood instead of a full 3x3 patch
                neighborhood = grayscale_image[i,
                               j - 1:j + 2]  # Adjust to a single row or column if AR model expects this

                # Check similarity to AR model
                if np.allclose(neighborhood[:len(ar_params)], ar_params, atol=20):
                    mask[i, j] = 255  # Mark as part of the water texture

        return mask

    def display_segmentation_result(self, original_image, result_mask, m):
        """
        Display the original image and result mask side by side.
        """
        # Convert PIL Image to NumPy array if necessary

        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # Plotting images and histogram
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            plt.imshow(original_image)
        else:
            plt.imshow(original_image, cmap='gray')
        plt.axis("off")

        # Result Mask
        plt.subplot(1, 3, 2)
        plt.title("Water Segmentation Mask")
        plt.imshow(result_mask, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def apply_mask(self, image, mask):
        pass
