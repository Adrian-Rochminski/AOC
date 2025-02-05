import os
from typing import List, Dict, Any, Union, Optional
from PIL import Image as PILImage
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries, slic
from Preprocessor import Preprocessor
from SingleSegmenter import SingleSegmenter
from Statistics import Statistics
import cv2

class ImageSegmentation:
    def __init__(self, data_loader: Any, class_names: Dict[str, int]):
        self.data_loader: Any = data_loader
        self.preprocessor: Preprocessor = Preprocessor()
        self.statistics: Statistics = Statistics()
        self.class_names: Dict[str, int] = class_names
        self.segmenters: List[SingleSegmenter] = []
        self.processed_images: List[PILImage.Image] = []
        self.processed_masks: List[Any] = []
        self.segmented_results: Dict[str, List[PILImage.Image]] = {}

    def load_data(self, mode: Union[str, int] = 'all') -> None:
        self.data_loader.load_data(mode=mode)
        self.statistics.update_total_images(len(self.data_loader))

    def resize_images_and_masks(self, images, masks, scale=0.5):
        resized_images = []
        resized_masks = []

        print(f"Resizing images and masks by scale {scale}...")
        for i, (img, mask) in enumerate(zip(images, masks)):
            img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            resized_images.append(img_resized)
            resized_masks.append(mask_resized)
        return resized_images, resized_masks

    def filter_images_with_class(self, images, masks, class_index, threshold=0.20):
        filtered_images = []
        filtered_masks = []
        print(f"Filtering images where class {class_index} covers ≥{threshold * 100}% of the area...")

        for i, (img, mask) in enumerate(zip(images, masks)):
            class_pixels = np.sum(mask == class_index)
            total_pixels = mask.size
            proportion = class_pixels / total_pixels
            if proportion >= threshold:
                filtered_images.append(img)
                filtered_masks.append(mask)
        print(f"Total filtered images: {len(filtered_images)}")
        return filtered_images, filtered_masks

    def train_with_visualization(self, num_of_images: int, n_segments=100, compactness=10):
        all_images = self.data_loader.images
        all_masks = self.data_loader.masks

        for class_name, choosen_group in self.class_names.items():
            path = f'{class_name}_segmenter.pkl'

            filtered_images, filtered_masks = self.filter_images_with_class(all_images, all_masks, choosen_group)
            resized_images, resized_masks = self.resize_images_and_masks(filtered_images, filtered_masks, scale=0.5)
            processed_img_array = resized_images[:num_of_images]
            processed_mask_array = resized_masks[:num_of_images]

            print(f"Using {len(processed_img_array)} images for segmentation.")
            print("Visualizing original images, masks, and superpixels...")
            for idx in range(min(2, len(processed_img_array))):
                img = processed_img_array[idx]
                mask = processed_mask_array[idx]

                segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)

                gt_mask = np.isin(mask, choosen_group)

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                axs[1].imshow(gt_mask, cmap='gray')
                axs[1].set_title(f'Ground Truth Mask ({class_name})')
                axs[1].axis('off')

                axs[2].imshow(mark_boundaries(img, segments))
                axs[2].set_title(f'Superpixels (n_segments={n_segments})')
                axs[2].axis('off')

                plt.show()

            segmenter = SingleSegmenter(class_name=class_name, class_indexes=choosen_group)

            features, labels = segmenter.prepare_data_with_superpixels(
                processed_img_array, processed_mask_array, n_segments=n_segments, compactness=compactness, augment=True
            )

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            if os.path.isfile(path):
                segmenter.load(path)
            else:
                segmenter.train(X_train, y_train)
            self.segmenters.append(segmenter)
            segmenter.save(path)

            segmenter.evaluate(
                X_test, y_test, processed_img_array, processed_mask_array, n_segments=n_segments, compactness=compactness
            )


    def display_statistics(self, num_examples: int = 5):
        stats: Dict[str, Any] = self.statistics.get_statistics()
        print("Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Processed images: {stats['processed_images']}")
        for class_name, results in stats['segmentation_results'].items():
            print(f"Segmented with {class_name}: {len(results)} images")

        print(f"\nDisplaying {num_examples} random examples of images, masks, and segmented results:")
        num_examples = min(num_examples, len(self.processed_images))
        random_indices: List[int] = random.sample(range(len(self.processed_images)), num_examples)

        for idx in random_indices:
            img: PILImage.Image = self.processed_images[idx]
            mask: Any = self.processed_masks[idx]

            num_cols: int = 2 + len(self.segmenters)
            fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
            axes[0].imshow(img)
            axes[0].set_title('Processed Image')
            axes[0].axis('off')
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Processed Mask')
            axes[1].axis('off')

            plt.show()

