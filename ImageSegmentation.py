import os
import pathlib
from typing import Any
import numpy as np
from skimage.color import rgb2gray
from skimage.util import img_as_float
from sklearn.exceptions import NotFittedError
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import cv2

from DataLoader import DataLoader
from FeaturesDataLoader import FeaturesDataloader
from SingleClassSegmenter import SingleClassSegmenter

CLASSES = ["water", "rocky", "land", "vegetation", "sky", "structures"]

class ImageSegmentation:
    def __init__(self, dataloader : DataLoader):
        self.dataloader = dataloader
        self.classifiers: {str, SingleClassSegmenter} = {}
        self.features_dataloader = FeaturesDataloader()

    def preprocess_data(self, scale: float = 0.5) -> None:
        self.features_dataloader.build(
            dataloader=self.dataloader,
            scale=scale
        )

    def train_classifiers(self, test_size: float = 0.2) -> None:
        print("\nTraining classifiers for all class groups:")
        for class_idx, class_name in enumerate(CLASSES):
            print(f"\nTraining {class_name} classifier...")

            try:
                features, labels = self.features_dataloader.get_class_data(class_idx)
            except ValueError as e:
                print(f"Skipping {class_name}: {e}")
                continue

            if len(features) == 0:
                print(f"No data for {class_name}, skipping.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )

            classifier = SingleClassSegmenter(class_idx)
            classifier.train(X_train, y_train)
            self.classifiers[class_name] = classifier
            print(f"Trained {class_name} classifier with {len(X_train)} samples")

    def predict_multi_class(self, img: np.ndarray, probability_threshold: float = 0.5) -> dict:
        img_float = img_as_float(img)
        segments = slic(img_float, n_segments=100, compactness=10)
        gabor_responses = self.features_dataloader.precompute_gabor_responses(rgb2gray(img))
        unique_segments = np.unique(segments)

        features = []
        valid_segment_ids = []
        for seg_id in unique_segments:
            feat = self.features_dataloader.extract_superpixel_features(img_float, segments, seg_id, gabor_responses)
            features.append(feat)
            valid_segment_ids.append(seg_id)

        features = np.array(features)
        features = self.features_dataloader.normalize_new_features(features)

        probabilities = {}
        for name, classifier in self.classifiers.items():
            try:
                proba = classifier.predict_proba(features)[:, 1]
                probabilities[name] = dict(zip(valid_segment_ids, proba))
            except NotFittedError:
                probabilities[name] = {seg_id: 0.0 for seg_id in valid_segment_ids}

        combined_mask = np.zeros_like(segments, dtype=int)
        for seg_id in unique_segments:
            class_probs = []
            for class_name in CLASSES:
                prob = probabilities[class_name].get(seg_id, 0.0)
                class_probs.append((class_name, prob))

            max_class, max_prob = max(class_probs, key=lambda x: x[1])
            if max_prob >= probability_threshold:
                combined_mask[segments == seg_id] = CLASSES.index(max_class) + 1

        return {
            'combined': combined_mask,
            'probabilities': probabilities,
            'segments': segments
        }

    def _postprocess_mask(self, predictions: np.ndarray, segments: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(segments, dtype=int)
        for seg_id, pred in enumerate(predictions):
            mask[segments == seg_id] = pred

        return self.classifiers['vegetation'].remove_isolated_predicted_segments(segments, mask)

    def _get_superpixel_adjacency(self, segments: np.ndarray) -> {int, set}:
        adjacency = {}
        height, width = segments.shape

        for y in range(height):
            for x in range(width):
                current = segments[y, x]
                if current not in adjacency:
                    adjacency[current] = set()

                if x + 1 < width and (right := segments[y, x + 1]) != current:
                    adjacency[current].add(right)
                    adjacency.setdefault(right, set()).add(current)

                if y + 1 < height and (bottom := segments[y + 1, x]) != current:
                    adjacency[current].add(bottom)
                    adjacency.setdefault(bottom, set()).add(current)

        return adjacency

    def _remove_isolated_segments(self, segments: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        adjacency = self._get_superpixel_adjacency(segments)
        refined_mask = pred_mask.copy()
        seg_labels = {seg: int(np.any(pred_mask[segments == seg])) for seg in np.unique(segments)}

        for seg_id, label in seg_labels.items():
            if label and not any(seg_labels[n] for n in adjacency[seg_id]):
                refined_mask[segments == seg_id] = 0

        return refined_mask

    def evaluate_multi_class(self, dataloader: DataLoader) -> {str, Any}:
        all_preds = []
        all_gts = []

        for img, true_mask in dataloader:
            results = self.predict_multi_class(img)
            pred_mask = results['combined']
            gt_resized = cv2.resize(true_mask, pred_mask.shape[::-1],
                                    interpolation=cv2.INTER_NEAREST)

            all_preds.append(pred_mask.flatten())
            all_gts.append(gt_resized.flatten())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_gts)

        metrics = {
            'overall_accuracy': accuracy_score(y_true, y_pred),
            'per_class_iou': jaccard_score(y_true, y_pred, average=None),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            # 'classification_report': classification_report(y_true, y_pred, target_names=CLASSES)
        }

        return metrics

    def save_classifiers(self, dir: str = "classifiers") -> None:
        for name, segmenter in self.classifiers.items():
            path = f'{dir}/{name}.pkl'
            segmenter.save(path)
        self.features_dataloader.save_normalization_params(f'{dir}/normalization_params.pkl')
        print(f"Saved all classifiers to {dir}")

    def load_classifiers(self, dir: str = "classifiers") -> None:
        self.features_dataloader.load_normalization_params(f'{dir}/normalization_params.pkl')
        for class_index, class_name in enumerate(CLASSES):
            path = f'{dir}/{class_name}.pkl'
            try:
                self.classifiers[class_name] = SingleClassSegmenter(class_index).load(path)
            except FileNotFoundError:
                print(f"Warning: Classifier for {class_name} not found")
        print(f"Loaded {len(self.classifiers)} classifiers")

    def visualize_results(self, img: np.ndarray, results: {str, np.ndarray},
                          ground_truth: np.ndarray = None) -> None:
        fig = plt.figure(figsize=(20, 10))

        ax1 = fig.add_subplot(241)
        ax1.imshow(img)
        ax1.set_title('Original Image')

        if ground_truth is not None:
            ax2 = fig.add_subplot(242)
            ax2.imshow(ground_truth)
            ax2.set_title('Ground Truth')

        ax3 = fig.add_subplot(243)
        ax3.imshow(results['combined'])
        ax3.set_title('Combined Prediction')

        for idx, (name, probs) in enumerate(results['probabilities'].items()):
            ax = fig.add_subplot(2, 4, idx + 4)
            prob_map = np.zeros_like(results['segments'], dtype=np.float32)
            for seg_id in np.unique(results['segments']):
                prob_map[results['segments'] == seg_id] = probs[seg_id]
            ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{name} Probability')
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()