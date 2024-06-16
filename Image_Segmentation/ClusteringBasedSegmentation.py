from scipy import ndimage
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def clustering_based_segmentation(image, k=3):
    """
    Clustering based image segmentation
    :param image: image to be segmented
    :param k: number of clusters
    :return: segmented image
    """
    pic = plt.imread(image) / 255  # dividing by 255 to bring the pixel values between 0 and 1
    pic_n = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])
    kmeans = KMeans(n_clusters=7, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    plt.imshow(cluster_pic)
    plt.show()


clustering_based_segmentation('1.jpeg')
