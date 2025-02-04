from matplotlib import pyplot as plt

from DataLoader import DataLoader
from ImageSegmentation import ImageSegmentation

loader = DataLoader()
loader.load_data(mode=2000)
test_loader = loader

train_loader, test_loader = loader.train_test_split(
    test_size=0.2,
    random_state=42,
    stratify_by='class'
)

print(f"Training images: {len(train_loader)}")
print(f"Test images: {len(test_loader)}")

# segmenter = ImageSegmentation(train_loader)
# segmenter.preprocess_data()
# segmenter.train_classifiers()
# segmenter.save_classifiers()

segmenter = ImageSegmentation(None)
segmenter.load_classifiers()

# results = segmenter.predict_multi_class(new_image)
# segmenter.visualize_results(new_image, results)

# metrics = segmenter.evaluate_multi_class(test_loader)
# print(metrics)

# Experiment with different probability thresholds

def show(img):
    plt.imshow(img)
    plt.axis('off')
    plt.colorbar()
    plt.show()

image, mask = test_loader[1]
print(mask)
show(image)
show(mask)
results = segmenter.predict_multi_class(image, probability_threshold=0.7)
segmenter.visualize_results(image, results, mask)
# print(results)
# show(results)
# results = segmenter.predict_multi_class(image, probability_threshold=0.5)
# show(results)