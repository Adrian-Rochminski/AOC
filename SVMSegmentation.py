import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


class RandomForestSegmentationWithSuperpixels:
    def __init__(self, class_name, class_indexes):
        self.class_name = class_name
        self.class_indexes = class_indexes
        self.clf = RandomForestClassifier(
            n_estimators=2000, class_weight='balanced_subsample', max_depth=30
        )
        print(
            f"Initialized RandomForestSegmentationWithSuperpixels for class group '{class_name}' with indexes {class_indexes}.")

    def extract_superpixel_features(self, img, segments, superpixel_id):
        mask_segment = segments == superpixel_id
        mean_color = img[mask_segment].mean(axis=0)
        if img.ndim == 3:
            gray_img = rgb2gray(img)
        else:
            gray_img = img
        lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
        lbp_hist = np.histogram(lbp[mask_segment], bins=np.arange(0, 11), density=True)[0]
        edge_map = sobel(gray_img)
        edge_density = edge_map[mask_segment].mean()
        coords = np.argwhere(mask_segment)
        centroid_x = coords[:, 1].mean() / img.shape[1]
        centroid_y = coords[:, 0].mean() / img.shape[0]
        color_hist = []
        for channel in range(img.shape[2]):
            hist, _ = np.histogram(img[..., channel][mask_segment], bins=16, range=(0, 1), density=True)
            color_hist.extend(hist)
        feature_vector = np.concatenate(
            [mean_color, lbp_hist, [edge_density, centroid_x, centroid_y], color_hist]
        )
        return feature_vector

    def augment_data(self, images, masks):
        augmented_images = []
        augmented_masks = []
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
        ], random_order=True)

        for img, mask in zip(images, masks):
            det = seq.to_deterministic()
            augmented_img = det.augment_image(img)
            augmented_mask = det.augment_image(mask)
            augmented_images.append(augmented_img)
            augmented_masks.append(augmented_mask)

        return augmented_images, augmented_masks

    def prepare_data_with_superpixels(self, images, masks, n_segments=100, compactness=10, augment=True):
        if augment:
            print("Applying data augmentation...")
            images, masks = self.augment_data(images, masks)
        else:
            print("Data augmentation not applied.")

        features_list = []
        labels_list = []
        print(f"Preparing data with superpixels for {len(images)} images and masks.")

        for idx, (img, mask) in enumerate(zip(images, masks)):
            print(f"Processing image {idx + 1}/{len(images)}...")
            img = img_as_float(img)

            segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
            unique_segments = np.unique(segments)
            print(f"  Number of superpixels: {len(unique_segments)}")

            for segment_id in unique_segments:
                feature_vector = self.extract_superpixel_features(img, segments, segment_id)
                features_list.append(feature_vector)
                mask_segment = segments == segment_id
                label = int(np.any(np.isin(mask[mask_segment], self.class_indexes)))
                labels_list.append(label)

        features = np.array(features_list)
        labels = np.array(labels_list)
        print(f"Total features shape: {features.shape}")
        print(f"Total labels shape: {labels.shape}")

        return features, labels

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict(X_test)
        return predictions

    def calculate_jaccard_index(self, gt_mask, pred_mask):
        gt_mask_int = gt_mask.astype(int).ravel()
        pred_mask_int = pred_mask.astype(int).ravel()
        return jaccard_score(gt_mask_int, pred_mask_int)

    def get_superpixel_adjacency(self, segments):
        adjacency_dict = {}
        height, width = segments.shape

        for y in range(height):
            for x in range(width):
                current_segment = segments[y, x]
                if current_segment not in adjacency_dict:
                    adjacency_dict[current_segment] = set()

                if x + 1 < width:
                    right_segment = segments[y, x + 1]
                    if right_segment != current_segment:
                        adjacency_dict[current_segment].add(right_segment)
                        if right_segment not in adjacency_dict:
                            adjacency_dict[right_segment] = set()
                        adjacency_dict[right_segment].add(current_segment)

                if y + 1 < height:
                    bottom_segment = segments[y + 1, x]
                    if bottom_segment != current_segment:
                        adjacency_dict[current_segment].add(bottom_segment)
                        if bottom_segment not in adjacency_dict:
                            adjacency_dict[bottom_segment] = set()
                        adjacency_dict[bottom_segment].add(current_segment)

        return adjacency_dict

    def remove_isolated_predicted_segments(self, segments, predicted_mask):
        adjacency_dict = self.get_superpixel_adjacency(segments)
        refined_pred_mask = predicted_mask.copy()
        unique_segments = np.unique(segments)
        segment_pred_labels = {}
        for seg_id in unique_segments:
            segment_pred_labels[seg_id] = int(np.any(predicted_mask[segments == seg_id]))
        for seg_id in unique_segments:
            if segment_pred_labels[seg_id] == 1:
                neighbors = adjacency_dict[seg_id]
                if not any(segment_pred_labels[n] == 1 for n in neighbors):
                    refined_pred_mask[segments == seg_id] = 0

        return refined_pred_mask

    def evaluate(self, X_test, y_test, test_images, test_masks, n_segments=100, compactness=10, remove_isolated=True):
        print("Starting evaluation...")
        predictions = self.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        print(cm)
        print(f"Visualizing predictions and computing Jaccard index for {len(test_images)} images.")
        jaccard_scores = []
        for idx, img in enumerate(test_images):
            img = img_as_float(img)
            segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
            unique_segments = np.unique(segments)
            features_list = []
            for segment_id in unique_segments:
                feature_vector = self.extract_superpixel_features(img, segments, segment_id)
                features_list.append(feature_vector)
            features = np.array(features_list)
            preds = self.predict(features)
            predicted_mask = np.zeros(segments.shape, dtype=int)
            for segment_id, pred in zip(unique_segments, preds):
                predicted_mask[segments == segment_id] = pred
            if remove_isolated:
                predicted_mask = self.remove_isolated_predicted_segments(segments, predicted_mask)
            gt_mask = np.isin(test_masks[idx], self.class_indexes)
            pred_mask = predicted_mask.astype(bool)
            jaccard_index = self.calculate_jaccard_index(gt_mask, pred_mask)
            jaccard_scores.append(jaccard_index)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title(f'Ground Truth Mask ({self.class_name})')
            axs[1].axis('off')

            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title(f'Predicted Mask ({self.class_name})\nJaccard: {jaccard_index:.4f}')
            axs[2].axis('off')
            plt.show()
        mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        print(f"Mean Jaccard Index (IoU) across test images: {mean_jaccard:.4f}")
        print("Evaluation completed.")