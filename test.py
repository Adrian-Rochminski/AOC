from ImageSegmentation import ImageSegmentation
from DataLoader import  DataLoader
#%%
classes = {
    "water": 0,
    "rocky": 1,
    "land": 2,
    "vegetation": 3,
    "sky": 4,
    "structures": 5
}

data_loader = DataLoader()
image_segmentator = ImageSegmentation(data_loader, classes)
#%%
image_segmentator.load_data(mode=6000)  # Options: 'all', 'half', 'quarter', or an integer
#%%
image_segmentator.data_loader.display_random_images_with_masks(2)
#%%
import datetime
start = datetime.datetime.now()
image_segmentator.train_with_visualization(3000, n_segments=100, compactness=10)
print("Time: " + str(datetime.datetime.now() - start))