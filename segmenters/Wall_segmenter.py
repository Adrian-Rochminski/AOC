from Segmenter import Segmenter

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
from skimage import color
from PIL import Image
import maxflow  # Graph Cut library for MRF optimization

class WallSegmenter(Segmenter):
    def __init__(self,class_info, num_labels=2, smoothing=2, neighborhood=4):
        super().__init__(class_info)
        self.num_labels = 2  # Number of labels/classes
        self.smoothing = smoothing  # Smoothing parameter for MRF
        self.neighborhood = neighborhood  # Neighborhood system (4 or 8)


    def segment(self, image):
        # Convert image to grayscale for processing
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            gray_image = color.rgb2gray(image)
        else:
            gray_image = image

            # Flatten image for GMM
        pixels = gray_image.flatten().reshape(-1, 1)

        # Fit GMM to separate water and non-water regions
        gmm = GaussianMixture(n_components=self.num_labels, covariance_type='tied')
        gmm.fit(pixels)
        labels = gmm.predict(pixels).reshape(gray_image.shape)

        # Initialize MRF graph using PyMaxflow
        height, width = gray_image.shape
        graph = maxflow.Graph[float]()
        node_ids = graph.add_nodes(height * width)

        # Unary potentials: Likelihood of pixel being water or not
        water_prob = gmm.predict_proba(pixels).reshape((height, width, self.num_labels))
        water_cost = -np.log(water_prob[..., 1] + 1e-10)
        non_water_cost = -np.log(water_prob[..., 0] + 1e-10)

        # Add edges for unary terms (data terms)
        for y in range(height):
            for x in range(width):
                node_id = y * width + x
                graph.add_tedge(node_id, water_cost[y, x], non_water_cost[y, x])

        # Pairwise potentials: Smoothness term to encourage neighboring pixels to have similar labels
        smoothness = self.smoothing
        for y in range(height):
            for x in range(width):
                node_id = y * width + x
                if x < width - 1:  # Right neighbor
                    neighbor_id = node_id + 1
                    graph.add_edge(node_id, neighbor_id, smoothness, smoothness)
                if y < height - 1:  # Bottom neighbor
                    neighbor_id = node_id + width
                    graph.add_edge(node_id, neighbor_id, smoothness, smoothness)

        # Find the minimum cut
        graph.maxflow()
        segmented_image = np.array([graph.get_segment(node) for node in range(height * width)]).reshape((height, width))
        mask = np.int32(segmented_image) * 255  # Convert boolean mask to 0-255 range for display

        return mask

    def display_segmentation_result(self, original_image, result_mask, m=255):
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
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
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
