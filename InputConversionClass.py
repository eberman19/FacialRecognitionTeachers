from PIL import Image
import numpy as np
from numpy import array
import os
from PCA import * 

class InputConversion:

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.matrix, self.shape = self.read_data()

    def jpg_image_to_array(self, image_path):
        #Loads JPEG image into 3D Numpy array of 1 row
##        with Image.open(image_path) as image:
##            im_arr1 = np.fromstring(image.tobytes(), dtype=np.uint8)
##        print("hello", im_arr1)
        
        img = Image.open(image_path)
        im_arr = array(img) #creates a matrix (L pixels by w pixels)
        Mim_arr = im_arr.reshape(im_arr.size,1) #1 column
        return im_arr, Mim_arr

    def array_to_jpg_image(self, image_vector):
        image_matrix = image_vector.reshape(self.shape)
        img = Image.fromarray(image_matrix)
        img.save("output.png")
        img.show()

    def read_data(self):
        IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), self.image_dir))
        all_images = []
        for fn in os.listdir(IMG_DIR):
            all_images.append(self.jpg_image_to_array(os.path.join(IMG_DIR, fn))[1]) #image path to folder imgs
            Overallshape = self.jpg_image_to_array(os.path.join(IMG_DIR, fn))[0].shape #records lxw of images
        all_images_matrix = np.asarray(all_images) #np convert list of sublists to matrix
        shape1,shape2,x = all_images_matrix.shape
        Mall_images_matrix = all_images_matrix.reshape(shape1,shape2).T #creates number of images by pixels so transpose to pixels by number of images
        return Mall_images_matrix, Overallshape

    def interactPCA(self):
        FR = PCA(self.matrix)
        v_reduce, z = FR.ProcessData(3)
        return v_reduce, z, FR.MagMean

Instance = InputConversion("imgs")
##wm, wmm = Instance.jpg_image_to_array("Waysek1 copy.jpg")
##print(wmm)
##Instance.array_to_jpg_image(wmm)
v, z, mm = Instance.interactPCA()
eigenfaces = [v[:,i] * mm/np.linalg.norm(v[:,i]) for i in range(v.shape[1])]
eigenfaces = [eigenfaces[i].astype(np.uint8) for i in range(len(eigenfaces))]
#print(eigenface1.astype(np.uint8))
for i in range(len(eigenfaces)):
    Instance.array_to_jpg_image(eigenfaces[i])

