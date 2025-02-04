import pickle
import random

import polars as pl
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.util import img_as_float
from tqdm import tqdm
import cv2
from skimage.segmentation import slic


class FeaturesDataloader:
    def __init__(self):
        self.df = pl.DataFrame()
        self.feature_columns = []
        self.feature_stats = {}
        self.feature_order = []

    def save_normalization_params(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'feature_columns': self.feature_columns,
                'feature_stats': self.feature_stats
            }, f)

    def load_normalization_params(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.feature_columns = data['feature_columns']
            self.feature_stats = data['feature_stats']

    def normalize_new_features(self, features: np.ndarray) -> np.ndarray:
        if not self.feature_stats:
            raise ValueError("No normalization parameters loaded")

        normalized = np.zeros_like(features)
        for idx, col in enumerate(self.feature_columns):
            min_val = self.feature_stats[col]['min']
            max_val = self.feature_stats[col]['max']

            if (max_val - min_val) == 0:
                normalized[:, idx] = 0.0
            else:
                normalized[:, idx] = (features[:, idx] - min_val) / (max_val - min_val)
        return normalized

    def build(self, dataloader, scale: float = 0.5, discard_random = True) -> None:
        data = []
        for img_idx, (img, mask) in enumerate(tqdm(dataloader, desc="Building features")):
            processed_img = img_as_float(self._resize_image(img, scale))
            processed_mask = self._resize_image(mask, scale, interpolation=cv2.INTER_NEAREST).astype(int)
            present_classes = np.unique(processed_mask).tolist()
            segments = slic(processed_img, n_segments=100, compactness=10)
            gabor_responses = self.precompute_gabor_responses(rgb2gray(processed_img))

            for seg_id in np.unique(segments):
                if discard_random and random.random() < 0.6:
                    continue
                features = self.extract_superpixel_features(processed_img, segments, seg_id, gabor_responses)
                mask_region = processed_mask[segments == seg_id]
                labels, counts = np.unique(mask_region, return_counts=True)
                majority_label = labels[np.argmax(counts)]

                row = {
                    "image_idx": img_idx,
                    "superpixel_id": int(seg_id),
                    "label": int(majority_label),
                    "present_classes": present_classes
                }
                for i, f in enumerate(features):
                    row[f"feature_{i}"] = float(f)
                data.append(row)

        self.df = pl.DataFrame(data)
        self.feature_columns = [col for col in self.df.columns if col.startswith("feature_")]
        self.feature_stats = {col: {} for col in self.feature_columns}
        self._normalize_features()
        self.df = self.df.with_columns(
            pl.concat_list(self.feature_columns).alias("features")
        ).drop(self.feature_columns)

        self.df.write_parquet("full_dataset.parquet")

    def _normalize_features(self) -> None:
        for col in self.feature_columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()

            self.feature_stats[col]['min'] = min_val
            self.feature_stats[col]['max'] = max_val

            if (max_val - min_val) == 0:
                self.df = self.df.with_columns(pl.lit(0.0).alias(col))
            else:
                self.df = self.df.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                )

    def get_normalization_params(self) -> dict:
        return {
            col: (self.feature_stats[col]['min'], self.feature_stats[col]['max'])
            for col in self.feature_columns
        }

    def get_class_data(self, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.df.is_empty():
            raise ValueError("FeaturesDataloader not built. Call build() first.")

        filtered = self.df.filter(
            pl.col("present_classes").list.contains(class_idx)
        )

        if filtered.is_empty():
            return np.array([]), np.array([])

        features = np.array(filtered["features"].to_list())
        labels = (filtered["label"] == class_idx).cast(pl.Int8).to_numpy()
        return features, labels

    def _resize_image(self, img: np.ndarray, scale: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)

    def extract_superpixel_features(self, img: np.ndarray, segments: np.ndarray,
                                    superpixel_id: int, gabor_responses) -> np.ndarray:
        mask = segments == superpixel_id
        gray_img = rgb2gray(img) if img.ndim == 3 else img
        color_features = self._compute_color_features(img, mask)
        lbp_features = self._compute_texture_features(gray_img, mask)
        gabor_features = [gabor_responses[key][mask].mean() for key in gabor_responses.keys()]
        centroid_x, centroid_y = self._compute_spatial_features(mask, img.shape[:2])

        return np.concatenate([
            color_features,
            lbp_features,
            gabor_features,
            [centroid_x, centroid_y]
        ])

    def precompute_gabor_responses(self, gray_img: np.ndarray, freqs=[0.1, 0.3], thetas=[0, np.pi / 4]):
        gabor_responses = {}
        for freq in freqs:
            for theta in thetas:
                filt_real, filt_imag = gabor(gray_img, frequency=freq, theta=theta)
                gabor_responses[(freq, theta)] = np.sqrt(filt_real ** 2 + filt_imag ** 2)
        return gabor_responses

    def _compute_texture_features(self, gray_img: np.ndarray, mask: np.ndarray) -> (np.ndarray, [float]):
        lbp = local_binary_pattern(np.uint8(gray_img * 255), P=8, R=1, method="uniform")
        lbp_values = lbp[mask]
        return [lbp_values.mean(), lbp_values.std()]

    def _compute_color_features(self, img: np.ndarray, mask: np.ndarray) -> (np.ndarray, [float]):
        pixels = img[mask]
        mean_color = pixels.mean(axis=0)
        std_color = pixels.std(axis=0)
        return np.concatenate([mean_color, std_color])

    def _compute_spatial_features(self, mask: np.ndarray, img_shape: (int, int)) -> (float, float):
            coords = np.argwhere(mask)
            centroid_x = coords[:, 1].mean() / img_shape[1]
            centroid_y = coords[:, 0].mean() / img_shape[0]
            return centroid_x, centroid_y
