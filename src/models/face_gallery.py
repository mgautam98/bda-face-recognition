import os
import numpy as np


# Face Gallery file name
gallery_file = 'face_gallery.npy'


def save_to_gallery(face_encodings):
  with open(gallery_file, 'wb') as gallery:
    np.save(gallery, face_encodings)

def load_gallery():
  with open(gallery_file, 'rb') as gallery:
    faces = np.load(gallery)
  return faces


# save_to_gallery(images)
# imgs = load_gallery()