from PIL import Image
import numpy as np

im = Image.open("Waysek1.jpg")
#im.rotate(45).show()

class InputConversion:

    def __init__(self, image):
        self.image = image

    def jpg_image_to_array(self, image_path):
        
        #Loads JPEG image into 3D Numpy array of shape
        with Image.open(image_path) as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        return im_arr

Waysek = InputConversion("Waysek1.jpg")
Waysek.jpg_image_to_array(Waysek.image)


