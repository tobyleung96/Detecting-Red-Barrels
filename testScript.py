import os
import math
import numpy as np
from skimage.measure import label, regionprops
from skimage import data, util
from cv2 import cv2
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt


def muCalculator(pointsArray):
    mu = np.mean(pointsArray, axis=0)
    return mu


def sigmaCovCalculator(pointsArray):
    mu = muCalculator(pointsArray)
    sigma = (1/pointsArray.shape[0]) * \
        np.dot((pointsArray - mu).T, (pointsArray - mu))
    return sigma


def gaussian(pointsArray, mu, sigma):
    gaussian = np.zeros((pointsArray.shape[0], 1))
    for pointIndex, point in enumerate(pointsArray):
        difference = (point-mu).T
        multiplier = 1 / (math.pow((2*math.pi), (3/2)) *
                          (math.pow(np.linalg.det(sigma), (1/2))))
        exponent = math.exp(-0.5 * np.dot(np.dot(difference.T,
                                                 np.linalg.inv(sigma)), difference))
        output = multiplier * exponent
        gaussian[pointIndex] = output
    return gaussian


def calculateGMMProbabilities(inputImage, clusters):
    GMMOutput = np.zeros((inputImage.shape[0], 1))
    for cluster in clusters:
        simpleGaussianOutput = gaussian(
            inputImage, cluster["mu"], cluster["sigma"])
        GMMOutput = GMMOutput + ((1/clusters.shape[0]) * simpleGaussianOutput)
    return GMMOutput


folder = "Test_Set"
clusters = np.load('clusters.npy', allow_pickle=True)
for filenameIndex, filename in enumerate(os.listdir(folder)):
    print("Calculating probabilities for file number: " +
          str(filenameIndex) + "...")
    # Reading images as RGB and HSV
    img_RGB = cv2.cvtColor(cv2.imread(os.path.join(
        folder, filename), 1), cv2.COLOR_BGR2RGB)
    img_HSV = cv2.cvtColor(cv2.imread(os.path.join(
        folder, filename), 1), cv2.COLOR_BGR2HSV)
    img_dim = img_RGB.shape[0]*img_RGB.shape[1]

    # make sure the input image is reshaped and normalized
    input_image = img_HSV.reshape(img_dim, 3)
    input_image = input_image[input_image.sum(axis=(1)) != 0]/255

    # do the same to RGB version for plotting later
    input_image_RGB = img_RGB.reshape(img_dim, 3)
    input_image_RGB = input_image_RGB[input_image_RGB.sum(axis=(1)) != 0]/255

    # calculate GMM probabilities
    GMMOutput = calculateGMMProbabilities(input_image, clusters)

    print("Conducting image filtering and computations...")
    # image erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    GMMOutput = np.float64(GMMOutput)
    GMMOutput = cv2.erode(GMMOutput, kernel, iterations=1)
    GMMOutput = cv2.dilate(GMMOutput, kernel, iterations=3)

    # replace image with binary (1,0) for white and black
    GMMOutput = np.where(GMMOutput > 0.99, 1, 0)

    # label image with skimage
    label_img = label(GMMOutput.reshape(
        img_RGB.shape[0], img_RGB.shape[1]), connectivity=2)

    # display results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(input_image_RGB.reshape(img_RGB.shape[0], img_RGB.shape[1], 3))
    for region in regionprops(label_img):
        if region.area >= 1200:
            bboxAspect = (region.bbox[2]-region.bbox[0]) / \
                (region.bbox[3]-region.bbox[1])
            if (bboxAspect > 1.0325 and bboxAspect < 1.9528):
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle(
                    (minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
                ratio = 493.0884217183345
                barrelDistance = str(
                    round(ratio / math.sqrt(region.area), 2)) + 'm'
                plt.annotate(
                    barrelDistance, (region.centroid[1], region.centroid[0]), color='white')
                plt.show()
                print("Test Image Number: " + str(filenameIndex) + ",  Centroid Row:  " + str(
                    region.centroid[1]) + ",  Centroid Column:  " + str(region.centroid[0]) + ",  Distance: " + str(barrelDistance))

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
