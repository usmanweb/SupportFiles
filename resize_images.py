import cv2
import os
import skimage.io
import numpy as np

#path_labels ='/home/mohbat/RoadSegmentation/DataSet/Mapillary/mapillary-vistas-dataset_public_v1.0/validation/instances/'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
i = 0
def read_files_list(path):
    file_names = []
    files = os.listdir(path)
    for i in files:
      file_names.append(path+i)
    return file_names

def resize_images(path_images):
  i = 0
  image_files =read_files_list(path_images)
  for image in image_files:
     if (i%100==0):
         print ('Image number:',i)
     im = cv2.imread(image)
     im = cv2.resize(im, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
     cv2.imwrite(image, im)
     i+=1

#test
#path_images = '/windows/D/DataSets for Image Segmentation/CityScapes/test/'
#resize_images (path_images)
#path_images = '/windows/D/DataSets for Image Segmentation/CityScapes/test_labels/'
#resize_images (path_images)


#train
path_images = 'lab/Extrovert/'
resize_images (path_images)
