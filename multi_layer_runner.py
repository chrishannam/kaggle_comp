import numpy as np
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import keras
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
import tensorflow as tf
from os.path import join, abspath, dirname
from functional import seq
from scipy.misc import imread
from collections import defaultdict
from scipy.misc import imread

SIZE = 64, 64


try:
    from atlas_kaggle.settings import data_dir
    print(f'Using {data_dir} for data')
except Exception as e:
    print('Using defaults for data_dir')
    data_dir = "/home/jessica/Kaggle/HumanProteinAtlasImageClassification/kaggle_comp/data"


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

def resize(image):
    return image.resize(size=SIZE)


def load_images(basepath, img_ids):
    print(basepath)
    images_matrix = np.array([load_image(basepath, img) for img in img_ids])
    return images_matrix


def load_image(basepath, image_id):
    """ Reads an image data by file id, and builds a matrix for this image
        - 4 channels for each filter colour
        - 512 * 512 for each image dimension
    :param basepath : path to image folder
    :param image_id : image name, minus the filter and png part
    :returns: image matrix """
    # empty numpy array of 4 (colour channel) by 512*512 (image width * height)
    image = np.zeros(shape=(4,512,512))
    image[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    image[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    image[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    image[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return image


def load_train_csv(gold_path):
    """ Loads train.csv in a dict of filename to list of gold classes labels dictionary
    :param gold_path: path to train.csv
    :type gold_path: str
    :returns: loaded gold data
    :rtype: dict of str:list of str """
    data = pd.read_csv(open(gold_path, "r"))
    data['Target'] = data['Target'].str.split(' ')  # making Target a list of labels
    return data


def load_training_dataset(training_csv, training_images_folder, batch_size):
    all_matrices = np.array([])
    img_to_label_df = load_train_csv(training_csv)
    ids = img_to_label_df.Id
    batches = [ids[x:x + batch_size] for x in range(0, ids.size, batch_size)]
    print(batches[:2])
    for batch in batches[:1]:
        print("loading images")
        b = load_images(training_images_folder, batch)
        print(b)
        all_matrices = np.concatenate(all_matrices, b)
    return all_matrices


def main():
    train_csv = f"{data_dir}/train.csv"
    train_imgs = f"{data_dir}/train/"
    test_imgs = f"{data_dir}/test"
    sample_submission = f"{data_dir}/sample_submission.csv"
    vgg_16 = VGG16()

    load_train_csv(f"{data_dir}/train.csv")
    load_training_dataset(train_csv, train_imgs, 10)
    pass




if __name__ == "__main__":
    # execute only if run as a script
    main()
