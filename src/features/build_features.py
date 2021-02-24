import os
import pywt
import pywt.data
import cv2
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

test_data = '/content/drive/MyDrive/research/converted/test_data'
train_data = '/content/drive/MyDrive/research/converted/train_data'
testing = '/content/drive/MyDrive/research/converted/testing_data'

def extract_n_concat(img):
  # DWT extract
  LL, (LH, HL, HH) = pywt.dwt2(img, 'bior1.3')
  # DCT extract
  imf = np.float32(img)
  dst = cv2.dct(imf)
  img = np.uint8(dst)

  # Concat
  a = np.concatenate([LL,LH], axis=0)
  b = np.concatenate([HL,img[:59, :59]/255.0], axis=0)
  c = np.concatenate([a, b], axis=1)
  return c

def load_images(path):
  files = os.listdir(path)
  images = []
  for file in tqdm(files, desc="Loading: "):
    im = io.imread(os.path.join(path, file))
    im_arr = np.array(im)
    im_arr = im_arr[:114, :114]
    im_arr = rgb2gray(im_arr)
    img_contat = extract_n_concat(im_arr)
    images.append(img_contat)
  return images

images = load_images(testing)