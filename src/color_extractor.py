import cv2
from sklearn.cluster import KMeans
import numpy as np

def extract_dominant_colors(image, k=5):
    img = cv2.resize(image, (64, 64))
    img = img.reshape((img.shape[0]*img.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    return kmeans.cluster_centers_
