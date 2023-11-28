#similarity index
# seberapa mirip gambar dengan yang dicompare
# semakin kecil index, semakin mirip
# semakin tinggi, semakin tidak mirip

import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
import re
import matplotlib.pyplot as plt


DATA_PATH = "data"

features = []

for file in os.listdir(DATA_PATH):
    img_path = DATA_PATH + "/" + file
    img_name = file.split(".")[0]

    img_bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    normalized_hist = cv2.normalize(hist, None)

    flat_hist = normalized_hist.flatten()
    print(flat_hist, flat_hist.shape)

    features.append((img_name, flat_hist))
