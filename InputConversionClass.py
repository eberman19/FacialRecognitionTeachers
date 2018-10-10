from PIL import Image
import numpy as np
from numpy import array
import os
from PCA import * 

class InputConversion(PCA):

    def __init__(self, image_dir, k):
        self.image_dir = image_dir
        self.Data, self.shape = self.read_data()
        self.SetSize = self.Data.shape[1]
        self.NormData, self.RowMean, self.MagMean = self.NormData()
        self.eigenfaces, self.coefficients = self.ProcessData(k)
        self.k = k

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

    def plot_eigenfaces(self):
        eigenfaces = [self.eigenfaces[:,i] * self.MagMean/np.linalg.norm(self.eigenfaces[:,i]) + self.RowMean for i in range(self.k)]
        eigenfaces = [eigenfaces[i].astype(np.uint8) for i in range(len(eigenfaces))]
        for i in range(len(eigenfaces)):
            self.array_to_jpg_image(eigenfaces[i])
        self.array_to_jpg_image(self.RowMean.astype(np.uint8)) #plots RowMean

Instance = InputConversion("imgs", 3)
Instance.plot_eigenfaces()


