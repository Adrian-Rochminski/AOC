from typing import List, Dict, Any, Union, Optional
from PIL import Image as PILImage
import random
import importlib
import matplotlib.pyplot as plt

from Preprocessor import Preprocessor
from Segmenter import Segmenter
from Statistics import Statistics


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
