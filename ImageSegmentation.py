from typing import List, Dict, Any, Union, Optional

import cv2
from PIL import Image as PILImage
import random
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Preprocessor import Preprocessor
from Segmenter import Segmenter
from Statistics import Statistics

import numpy as np
from matplotlib import pyplot as plt
from thundersvm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from skimage.color import rgb2gray



class SVMSegmentation:
    def __init__(self, class_name, class_index):
        self.class_name = class_name
        self.class_index = class_index
        self.svm_model = SVC(probability=True, max_iter=10)#, kernel='linear')

    def prepare_data1(self, images, masks):
        features = []
        labels = []
        i = 0
        for img, mask in zip(images, masks):
            # Convert to grayscale for simplicity
            gray_img = rgb2gray(np.array(img))

            # Flatten the image and mask
            img_flat = gray_img.flatten()
            mask_flat = mask.flatten()

            # Generate labels for the given class index
            binary_labels = (mask_flat == self.class_index).astype(int)

            features.append(img_flat)
            labels.append(binary_labels)
            print(i)
            print(len(features))
            print(len(labels))
            i+=1
        features = np.vstack(features)
        labels = np.hstack(labels)
        return features, labels

    def preprocess(self, image):
        gamma = 2.2
        gamma_correction = np.array(255 * (image / 255) ** (1 / gamma), dtype='uint8')
        pre_image = cv2.cvtColor(gamma_correction, cv2.COLOR_BGR2GRAY)
        pre_image = cv2.resize(pre_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        pre_image = cv2.Laplacian(pre_image, cv2.CV_64F)
        pre_image = cv2.equalizeHist(cv2.convertScaleAbs(pre_image))
        return pre_image

    def prepare_data(self, images, masks):
        # Convert all images to grayscale and flatten
        #images_array = [rgb2gray(np.array(img)).flatten() for img in images]
        images_array = [self.preprocess(np.array(img)).flatten() for img in images]
        images_flat = np.concatenate(images_array)  # Shape (n_images * n_pixels,)

        # Flatten masks
        masks_array = [cv2.resize(mask, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA).flatten() for mask in masks]
        masks_flat = np.concatenate(masks_array)  # Shape (n_images * n_pixels,)

        # Generate binary labels
        binary_labels = (masks_flat == self.class_index).astype(int)  # Shape (n_samples,)

        # Reshape features to have one column (since we have one feature per pixel)
        features = images_flat.reshape(-1, 1)  # Shape (n_samples, n_features)

        return features, binary_labels

    def train(self, features, labels):
        print("Training SVM...")
        self.svm_model.fit(features, labels)
        print("SVM training completed.")

    def predict(self, images):
        predictions = []
        for img in images:
            # Convert to grayscale
            # gray_img = rgb2gray(np.array(img))
            img_flat = img.flatten().reshape(1, -1)
            prediction = self.svm_model.predict(img_flat)
            predictions.append(prediction.reshape(img.shape))
        return predictions

    def evaluate(self, features, labels, test_images, test_masks):
        print("Evaluating SVM...")
        predictions = self.svm_model.predict(features)

        # Generate classification report
        print("Classification Report:")
        print(classification_report(labels, predictions))

        # Generate confusion matrix
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix for Class: {self.class_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Show random results
        self.visualize_results(test_images, test_masks, predictions)

    def visualize_results(self, images, masks, predictions):
        print("\nVisualizing results...")
        num_examples = 5
        indices = random.sample(range(len(images)), k=num_examples)

        for idx in indices:
            original_img = np.array(images[idx])
            expected_mask = masks[idx]
            predicted_mask = predictions[idx]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(expected_mask, cmap="gray")
            axes[1].set_title("Expected Mask")
            axes[1].axis("off")

            print(np.array(predicted_mask).shape)
            print(predicted_mask)
            axes[2].imshow(np.array(predicted_mask), cmap="gray")
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            plt.show()

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
                result: PILImage.Image = segmenter.segment(processed_img_pil, processed_mask_array)
                if segmenter.class_name not in self.segmented_results:
                    self.segmented_results[segmenter.class_name] = []
                self.segmented_results[segmenter.class_name].append(result)
                self.statistics.add_segmentation_result(segmenter.class_name, result)

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

    def svm_segmentation(self, num_of_images: int):
        processed_img_array: Any = self.data_loader.images[:num_of_images]
        processed_mask_array: Any = self.data_loader.masks[:num_of_images]
        water_segmenter = SVMSegmentation(class_name="Water", class_index=8)
        features, labels = water_segmenter.prepare_data(processed_img_array, processed_mask_array)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        water_segmenter.train(X_train, y_train)
        predictions = water_segmenter.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, np.array(predictions).flatten()))
        water_segmenter.evaluate(X_test, y_test, processed_img_array, processed_mask_array)
