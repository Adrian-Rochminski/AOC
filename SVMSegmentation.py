import numpy as np
from thundersvm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
class SVMSegmentation:
    def __init__(self, class_name, class_index):
        """
        Initializes the SVM segmenter for a specific class.

        Parameters:
        - class_name: Name of the class to segment (e.g., "Water")
        - class_index: The index representing the class in the mask (e.g., 8)
        """
        self.class_name = class_name
        self.class_index = class_index
        self.clf = SVC(max_iter=100)
        self.scaler = None  # Will be used if feature scaling is applied
        print(f"Initialized SVMSegmentation for class '{class_name}' with index {class_index}.")

    def prepare_data_with_filters(self, images, masks):
        """
        Prepares features and labels from images and masks.

        Parameters:
        - images: List of images (height x width x channels)
        - masks: List of corresponding masks (height x width)

        Returns:
        - features: Array of pixel features
        - labels: Array of labels (1 if pixel is of the class, else 0)
        """
        features_list = []
        labels_list = []
        print(f"Preparing data with {len(images)} images and masks.")

        for idx, (img, mask) in enumerate(zip(images, masks)):
            print(f"Processing image {idx + 1}/{len(images)}:")
            height, width, channels = img.shape
            print(f"  Image dimensions: {height}x{width}, Channels: {channels}")
            print(f"  Mask dimensions: {mask.shape}")

            img_flat = img.reshape(-1, channels)
            mask_flat = mask.flatten()
            print(f"  Flattened image shape: {img_flat.shape}")
            print(f"  Flattened mask shape: {mask_flat.shape}")

            features = img_flat
            labels = (mask_flat == self.class_index).astype(int)
            print(f"  Number of target class pixels: {np.sum(labels)}")
            print(f"  Total pixels: {len(labels)}")

            features_list.append(features)
            labels_list.append(labels)

        # Concatenate all features and labels
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)
        print(f"Total features shape: {features.shape}")
        print(f"Total labels shape: {labels.shape}")

        return features, labels

    def train(self, X_train, y_train):
        """
        Trains the SVM classifier.

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        """
        print("Starting training...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")

        self.clf.fit(X_train, y_train)
        print("Training completed.")

    def predict(self, X_test):
        """
        Predicts the labels for test data.

        Parameters:
        - X_test: Test features

        Returns:
        - predictions: Predicted labels
        """
        print("Starting prediction...")
        print(f"Test data shape: {X_test.shape}")

        predictions = self.clf.predict(X_test)
        print(f"Prediction completed. Predictions shape: {predictions.shape}")
        return predictions

    def evaluate(self, X_test, y_test, test_images, test_masks):
        """
        Evaluates the model and visualizes the results.

        Parameters:
        - X_test: Test features
        - y_test: Test labels
        - test_images: Original test images for visualization
        - test_masks: Original test masks for visualization
        """
        print("Starting evaluation...")
        predictions = self.predict(X_test)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Visualize predictions on a few test images
        num_images_to_display = ( len(test_images))  # Adjust as needed
        print(f"Visualizing results for {num_images_to_display} images.")
        i = 0
        for idx in range(num_images_to_display):
            print(f"Visualizing image {idx + 1}/{num_images_to_display}...")
            img = test_images[idx]
            mask = test_masks[idx]
            height, width, channels = img.shape
            print(f"  Image dimensions: {height}x{width}, Channels: {channels}")
            img_flat = img.reshape(-1, channels)

            img_pred_flat = self.predict(img_flat)
            img_pred_mask = img_pred_flat.reshape(height, width)
            class_mask = (mask == self.class_index).astype(int)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(img)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(class_mask, cmap='gray')
            axs[1].set_title(f'Ground Truth Mask ({self.class_name})')
            axs[1].axis('off')

            axs[2].imshow(img_pred_mask, cmap='gray')
            axs[2].set_title(f'Predicted Mask ({self.class_name})')
            axs[2].axis('off')
            i+=1
            plt.show()
            if i == 100:
                break

        print("Evaluation completed.")
