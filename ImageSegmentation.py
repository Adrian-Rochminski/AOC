from typing import List, Dict, Any, Union, Optional
from PIL import Image as PILImage
import random
import importlib
import matplotlib.pyplot as plt
import numpy as np
from Preprocessor import Preprocessor
from SVMSegmentation import SVMSegmentation
from Segmenter import Segmenter
from Statistics import Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2

class ImageSegmentation:
    def __init__(self, data_loader: Any, class_names: List[Dict[str, Any]]):
        self.data_loader: Any = data_loader
        self.preprocessor: Preprocessor = Preprocessor()
        self.statistics: Statistics = Statistics()
        self.class_names: List[Dict[str, Any]] = class_names
        self.segmenters: List[Segmenter] = []
        self.processed_images: List[PILImage.Image] = []
        self.processed_masks: List[Any] = []
        self.segmented_results: Dict[str, List[PILImage.Image]] = {}

    def select_classes(self, class_names_to_analyze: List[str], segmenter_folder: str = "segmenters") -> None:
        for class_name in class_names_to_analyze:
            class_info: Optional[Dict[str, Any]] = next(
                (c for c in self.class_names if c['name'].lower() == class_name.lower()), None)
            if class_info is None:
                raise ValueError(f"Class '{class_name}' not found in class names.")

            module_name: str = f"{segmenter_folder}.{class_name}_segmenter"
            segmenter_class_name: str = f"{class_name}Segmenter"

            try:
                module = importlib.import_module(module_name)
                segmenter_class = getattr(module, segmenter_class_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"No segmenter class found for '{class_name}'. Error: {e}")

            segmenter_instance: Segmenter = segmenter_class(class_info)
            self.segmenters.append(segmenter_instance)
            print(f"Selected class '{class_info['name']}' with class index {class_info['classIndex']}")

    def load_data(self, mode: Union[str, int] = 'all') -> None:
        self.data_loader.load_data(mode=mode)
        self.statistics.update_total_images(len(self.data_loader))

    def process_data(self, scale: float = 0.5) -> None:
        if not self.segmenters:
            raise ValueError("No classes selected for segmentation. Use select_classes() to choose classes.")
        for idx, (img_array, mask_array) in enumerate(self.data_loader):
            processed_img_array: Any = self.preprocessor.center_crop_array(img_array, scale=scale)
            processed_mask_array: Any = self.preprocessor.center_crop_mask(mask_array, scale=scale)
            processed_img_pil: PILImage.Image = PILImage.fromarray(processed_img_array.astype('uint8'))
            self.processed_images.append(processed_img_pil)
            self.processed_masks.append(processed_mask_array)
            self.statistics.increment_processed_images()
            for segmenter in self.segmenters:
                result: PILImage.Image = segmenter.segment(processed_img_pil)
                if segmenter.class_name not in self.segmented_results:
                    self.segmented_results[segmenter.class_name] = []
                self.segmented_results[segmenter.class_name].append(result)
                self.statistics.add_segmentation_result(segmenter.class_name, result)
                segmenter.display_segmentation_result(processed_img_pil, processed_mask_array, result)

    # def svm_segmentation(self, num_of_images: int):
    #     processed_img_array: Any = self.data_loader.images[:num_of_images]
    #     processed_mask_array: Any = self.data_loader.masks[:num_of_images]
    #     water_segmenter = SVMSegmentation(class_name="Water", class_index=8)
    #     features, labels = water_segmenter.prepare_data_with_filters(processed_img_array, processed_mask_array)
    #     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    #     water_segmenter.train(X_train, y_train)
    #     predictions = water_segmenter.predict(X_test)
    #     print("Classification Report:")
    #     print(classification_report(y_test, predictions.flatten()))
    #     water_segmenter.evaluate(X_test, y_test, processed_img_array, processed_mask_array)
    def resize_images_and_masks(self, images, masks, scale=0.5):
        """
        Resizes images and masks by a given scale.

        Parameters:
        - images: List of images (height x width x channels)
        - masks: List of corresponding masks (height x width)
        - scale: Scaling factor (e.g., 0.25 for 1/4th size)

        Returns:
        - resized_images: List of resized images
        - resized_masks: List of resized masks
        """
        resized_images = []
        resized_masks = []

        print(f"Resizing images and masks by scale {scale}...")
        for i, (img, mask) in enumerate(zip(images, masks)):
            # Resize image
            img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # Resize mask
            mask_resized = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            resized_images.append(img_resized)
            resized_masks.append(mask_resized)

            print(f"Image {i + 1}: Original size {img.shape[:2]} -> Resized size {img_resized.shape[:2]}")

        return resized_images, resized_masks

    def filter_images_with_class(self, images, masks, class_index):
        """
        Filters images and masks to include only those where the mask contains the specified class index.

        Parameters:
        - images: List of images (height x width x channels)
        - masks: List of corresponding masks (height x width)
        - class_index: The class index to filter by (e.g., 8 for water)

        Returns:
        - filtered_images: List of images containing the specified class
        - filtered_masks: List of corresponding masks
        """
        filtered_images = []
        filtered_masks = []

        print(f"Filtering images for class index {class_index}...")
        for i, (img, mask) in enumerate(zip(images, masks)):
            if np.any(mask == class_index):
                filtered_images.append(img)
                filtered_masks.append(mask)
            else:
                print(f"Image {i + 1} excluded: No pixels with class index {class_index}.")

        print(f"Total filtered images: {len(filtered_images)}")
        return filtered_images, filtered_masks

    def svm_segmentation(self, num_of_images: int):
        # Load images and masks
        all_images = self.data_loader.images
        all_masks = self.data_loader.masks

        # Filter images and masks to include only those with the target class
        filtered_images, filtered_masks = self.filter_images_with_class(all_images, all_masks, class_index=8)

        # Filter images and masks to include only those with the target class
        # Resize images and masks to 1/4th size
        resized_images, resized_masks = self.resize_images_and_masks(filtered_images, filtered_masks, scale=0.5)

        # Use only the first num_of_images after filtering and resizing
        processed_img_array = resized_images[:num_of_images]
        processed_mask_array = resized_masks[:num_of_images]

        print(f"Using {len(processed_img_array)} images for segmentation.")

        # Instantiate the SVMSegmentation for the "Water" class
        water_segmenter = SVMSegmentation(class_name="Water", class_index=8)

        # Prepare data
        features, labels = water_segmenter.prepare_data_with_filters(processed_img_array, processed_mask_array)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train the model
        water_segmenter.train(X_train, y_train)

        # Evaluate the model
        water_segmenter.evaluate(X_test, y_test, processed_img_array, processed_mask_array)

    def display_statistics(self, num_examples: int = 5) -> None:
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

