# Similarity index
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

# for file in os.listdir(DATA_PATH):
#     img_path = DATA_PATH + "/" + file
#     img_name = file.split(".")[0]

#     img_bgr = cv2.imread(img_path)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
#     normalized_hist = cv2.normalize(hist, None)

#     flat_hist = normalized_hist.flatten()
#     print(flat_hist, flat_hist.shape)

#     features.append((img_name, flat_hist))


# Create a function for extracting feature
def get_image_feature(path):
    img_bgr = cv2.imread(path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # comparing image in gray format
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])

    # comparing image in color format (it might be more accurate)
    # hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    normalized_hist = cv2.normalize(hist, None)

    flat_hist = normalized_hist.flatten()
    return flat_hist


# for DATA folder
for file in os.listdir(DATA_PATH):
    img_path = DATA_PATH + "/" + file
    img_name = file.split(".")[0]

    feature = get_image_feature(img_path)
    features.append((img_name, feature))

# for TARGET folder
TEST_PATH = 'target\kitkat-1.png'
test_feature = get_image_feature(TEST_PATH)

# Create a list for showing the result of distance difference
results = []
for name, feature in features:
    distance = euclidean(feature, test_feature)
    results.append((distance, name))

# Sorting the distance difference
# the smaller the value, the closer the distance
# the bigger the value, the further the distance
results = sorted(results)


# Print the image from the closest distance first
for distance, name in results:
    print(f"{name}: {distance}")
    # print(f"{name}: { round(100 - (distance * 100), 2) }%")


# Creatiing a plot for showing the top 3 most similar image (manually)
fig = plt.figure()

plt.subplot(3, 3, 1)
img_rgb = cv2.imread("data\kitkat-1.jpg")
warna = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
plt.imshow(warna)

plt.subplot(3, 3, 2)
img_rgb = cv2.imread("data\kitkat-2.jpg")
warna = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
plt.imshow(warna)


plt.subplot(3, 3, 3)
img_rgb = cv2.imread("data\silverqueen-3.png")
warna = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
plt.imshow(warna)

plt.show()