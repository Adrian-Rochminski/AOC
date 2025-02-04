import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import matplotlib.pyplot as plt


class SingleClassSegmenter:
    def __init__(self, class_index: int, feature_order: list = None):
        self.class_index = class_index
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1,
        )
        self.feature_order = feature_order

    def train(self, X_train, y_train):
        print(f"Training model with {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0, 2))
        return self.model.predict_proba(features)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def save(self, path):
        with open(path, 'wb+') as f:
            pickle.dump({
            'clf': self.model,
            'feature_order': self.feature_order
        }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['clf']
        self.feature_order = data['feature_order']
        return self

    def evaluate(
            self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            test_images: [np.ndarray],
            test_masks: [np.ndarray],
            n_segments: int = 100,
            compactness: int = 10,
            remove_isolated: bool = True
    ) -> float:
        print("Starting evaluation...")

        predictions = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        print(f"\nProcessing {len(test_images)} test images:")
        jaccard_scores = []

        for idx, (img, true_mask) in enumerate(zip(test_images, test_masks)):
            img_float = img_as_float(img)
            segments = slic(img_float, n_segments=n_segments, compactness=compactness)

            features = [self.extract_superpixel_features(img_float, segments, seg)
                        for seg in np.unique(segments)]
            preds = self.model.predict(np.array(features))

            pred_mask = np.zeros_like(segments, dtype=int)
            for seg_id, pred in zip(np.unique(segments), preds):
                pred_mask[segments == seg_id] = pred

            # if remove_isolated:
            #     pred_mask = self._remove_isolated_segments(segments, pred_mask)

            gt_mask = true_mask == self.class_index
            iou = jaccard_score(gt_mask.ravel(), pred_mask.ravel())
            jaccard_scores.append(iou)

            self._plot_results(img_float, gt_mask, pred_mask, iou, idx + 1)

        mean_iou = np.mean(jaccard_scores)
        print(f"\nMean Jaccard Index: {mean_iou:.4f}")
        return mean_iou

    def _plot_results(self, img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, iou: float, idx: int) -> None:
        print(f"Image {idx} - Jaccard: {iou:.4f}")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        titles = [
            'Original Image',
            f'Ground Truth ({self.class_index})',
            f'Predicted Mask (IoU: {iou:.3f})'
        ]

        for axis, data, title in zip(ax, [img, gt_mask, pred_mask], titles):
            axis.imshow(data, cmap='gray' if title != 'Original Image' else None)
            axis.set_title(title)
            axis.axis('off')

        plt.tight_layout()
        plt.show()