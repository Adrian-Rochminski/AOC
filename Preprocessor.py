class Preprocessor:
    def __init__(self):
        pass

    def center_crop(self, image, scale=0.5):
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        return image.crop((left, top, right, bottom))

    def center_crop_mask(self, mask, scale=0.5):
        height, width = mask.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return mask[top:bottom, left:right]

    def center_crop_array(self, array, scale=0.5):
        height, width = array.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return array[top:bottom, left:right]