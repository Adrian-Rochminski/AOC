from random import random


class Statistics:
    def __init__(self):
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'segmentation_results': {}
        }
        self.processed_images = []

    def update_total_images(self, count):
        self.stats['total_images'] = count

    def increment_processed_images(self):
        self.stats['processed_images'] += 1

    def add_segmentation_result(self, class_name, result):
        if class_name not in self.stats['segmentation_results']:
            self.stats['segmentation_results'][class_name] = []
        self.stats['segmentation_results'][class_name].append(result)

    def get_statistics(self):
        return self.stats

    def display_random_images(self, num_images=5):
        if len(self.processed_images) == 0:
            print("No images to display.")
            return
        num_images = min(num_images, len(self.processed_images))
        random_images = random.sample(self.processed_images, num_images)
        for idx, img in enumerate(random_images):
            print(f"Displaying image {idx + 1}/{num_images}")
            img.show()