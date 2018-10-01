#! /usr/bin/python3
from src.loader import Imager
from imutils import paths
import numpy as np
import pickle
import cv2
from sklearn import preprocessing

from keras.utils import np_utils


def loader(path):
    imPs = []
    imagePaths = list(paths.list_images(path))
    for (i, imP) in enumerate(imagePaths):
        imPs.append(imP)
    return imPs


def dataModel():
    '''
    load image preprocessed form pickel
    '''
    train = list()
    data = pickle.load(open("../dataModel-15-66.pkl", "rb"))
    for table in range(len(data)):  # 16
        for frame in range(len(data[table])):  # 66
            if frame % 11 is not 0:  # hu frame cuoi
                # print(table , " " , frame)
                data[table][frame] = cv2.resize(data[table][frame], (128, 64))
                train.append(data[table][frame])

    train = np.array(train)

    return train


def encodeLabel():
    le = preprocessing.LabelEncoder()
    _, label = readImage()
    labels = le.fit(list(label))
    labels = le.transform(list(label))
    labels = np.repeat(labels, 60)
    return (le, labels)


def label():
    """
            encode label
            return label to int
    """
    le, labels = encodeLabel()
    return labels


def labelValue():
    le, labels = encodeLabel()
    return le


def readImage() ->list():
    """
    read and preprocess image to frames
    a frame is a sign
    return list(frame)
    """
    print("read Image")
    im = Imager()
    imPaths = loader("../hinh")
    # print (imPaths)
    newPath = '../hinhCrop'
    noneBorder = '/home/banhtrung/Code/NLCS/hinhNoBorder'
    # for img in imPaths:
    # 	# print("read Image")
    # 	im.cropImage(img,newPath)

    # imPaths = loader(newPath)
    # for img in imPaths :
    #     im.noneBorder(img,noneBorder)

    dataTrain = list()
    labels = list()
    for path in loader(noneBorder):
        print(" {0}".format(path[len(noneBorder)+1:len(path)-4]))
        labels.append(path[len(noneBorder)+1:len(path)-4])
        dataTrain.append(im.frameImage(path))

    """
    pickle.dump(data,open("dataModel-15-66.pkl","wb"))
  """
    return (dataTrain, labels)


def train_test_data():
    """
            it return a tuple-list with (data,label)
    """
    return list(zip(dataModel(), label()))


def noneBorder(image):
    """
      del all bolder if red = green = blue < 255
    """
    white = [255, 255, 255]
    for w in range(len(image[0])):
        for h in range(len(image)):
            r, g, b = image[h, w]
            if r == g and g == b and r < 255:
                image[h, w] = white
            if (r < 80 and g < 100) or (b < 50 and g < 100):
                image[h, w] = white

    return image
