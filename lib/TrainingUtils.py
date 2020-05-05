"""
These are various mehtods that can be used in the training process.  Some
return values, some display images.  Best used in Jupyter Notebooks.
"""

from __future__ import division, print_function, absolute_import
# Use one of these based on the version of skimage loaded
from skimage.util import montage
# from skimage.util.montage import montage2d as montage

import os
from random import sample
import numpy as np
import csv
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report, \
                            accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tqdm import tqdm

##############################################################


class GlobalParms(object):
    """
    Class that contains global variables.  In other notebooks you will see
    global vars as upper case variables, dictionaries or classes.  I picked
    a modified class approach because I felt it gave the best balance between
    structure and flexibility.

    These are all possible global vars, most notebooks will only use a subset.
    """
    def __init__(self, **kwargs):
        self.keys_and_defaults = {
         "MODEL_NAME": "",  # if you leave .h5 off, puts into a subdirectory
         "ROOT_PATH": "",  # Location of the data for storing any data or files
         "TRAIN_DIR": "",  # Subdirectory in the Root for Training files
         "TEST_DIR": "",  # Optional subdirectory in  Root for Testing file
         "SUBMISSION_PATH": None,  # Optional subdirectory for Contest files
         "MODEL_PATH": None,  # Optional, subdirectory for saving/loading model
         "TRAIN_PATH": None,  # Subdirectory in the Root for Training files
         "TEST_PATH": None,  # Optional subdirectory in  Root for Testing file
         "SMALL_RUN": False,   # Optional, run size will be reduced
         "NUM_CLASSES": 0,  # Number of classes
         "CLASS_NAMES": [],  # list of class names
         "IMAGE_ROWS": 0,  # Row size of the image
         "IMAGE_COLS": 0,  # Col size of the image
         "IMAGE_CHANNELS": 0,  # Num of Channels, 1 for Greyscale, 3 for color
         "BATCH_SIZE": 0,  # Number of images in each batch
         "EPOCS": 0,  # Max number of training EPOCS
         "ROW_SCALE_FACTOR": 1,  # Optional, allows scaling of an image.
         "COL_SCALE_FACTOR": 1,  # Optional, allows scaling of an image.
         "IMAGE_EXT": ".jpg",  # Extent of the image file_ext
         # Optional, default is np.float64, reduce memory by using np.float32
         # or np.float16
         "IMAGE_DTYPE": np.float32,
         # Optional, change default if needed, can save memory space
         "Y_DTYPE": np.int,
         "LOAD_MODEL": False,  # Optional, If you want to load a saved model
         "SUBMISSION": "submission.csv",  # Optional, Mainly used for Kaggle
         "METRICS": ['accuracy'],  # ['categorical_accuracy'], ['accuracy']
         "FINAL_ACTIVATION": 'sigmoid',  # sigmoid, softmax
         "LOSS": ""  # 'binary_crossentropy', 'categorical_crossentropy'
        }

        self.__dict__.update(self.keys_and_defaults)
        self.__dict__.update((k, v) for k, v in kwargs.items()
                             if k in self.keys_and_defaults)

        # Automatically reduce the training parms, change as needed
        if self.__dict__["SMALL_RUN"]:
            self.__dict__["BATCH_SIZE"] = 1
            self.__dict__["EPOCS"] = 2
            self.__dict__["ROW_SCALE_FACTOR"] = 1
            self.__dict__["COL_SCALE_FACTOR"] = 1

        # Use configuration items to create real ones
        self.__dict__["SCALED_ROW_DIM"] = \
            np.int(self.__dict__["IMAGE_ROWS"] /
                   self.__dict__["ROW_SCALE_FACTOR"])

        self.__dict__["SCALED_COL_DIM"] =  \
            np.int(self.__dict__["IMAGE_COLS"] /
                   self.__dict__["COL_SCALE_FACTOR"])

        if self.__dict__["TRAIN_PATH"] is None:  # Not passed, so set it
            self.__dict__["TRAIN_PATH"] = \
                os.path.join(self.__dict__["ROOT_PATH"],
                             self.__dict__["TRAIN_DIR"])

        if self.__dict__["TEST_PATH"] is None:  # Not passed, so set it
            self.__dict__["TEST_PATH"] = \
                os.path.join(self.__dict__["ROOT_PATH"],
                             self.__dict__["TEST_DIR"])

        if self.__dict__["SUBMISSION_PATH"] is None:  # Not passed, so set
            self.__dict__["SUBMISSION_PATH"] = \
                os.path.join(self.__dict__["ROOT_PATH"],
                             self.__dict__["SUBMISSION"])
        else:
            self.__dict__["SUBMISSION_PATH"] = \
                os.path.join(self.__dict__["SUBMISSION_PATH"],
                             self.__dict__["SUBMISSION"])

        if self.__dict__["MODEL_PATH"] is None:  # Not passed, so set it
            self.__dict__["MODEL_PATH"] = \
                os.path.join(self.__dict__["ROOT_PATH"],
                             self.__dict__["MODEL_NAME"])
        else:
            self.__dict__["MODEL_PATH"] = \
                os.path.join(self.__dict__["MODEL_PATH"],
                             self.__dict__["MODEL_NAME"])

        self.__dict__["IMAGE_DIM"] = \
            (self.__dict__["SCALED_ROW_DIM"],
             self.__dict__["SCALED_COL_DIM"],
             self.__dict__["IMAGE_CHANNELS"])

        if self.__dict__["IMAGE_CHANNELS"] == 1:
            self.__dict__["COLOR_MODE"] = "grayscale"
        else:
            self.__dict__["COLOR_MODE"] = "rgb"

    def set_train_path(self, train_path):
        self.__dict__["TRAIN_PATH"] = train_path

    def set_class_names(self, class_name_list):
        self.__dict__["CLASS_NAMES"] = class_name_list

        if self.__dict__["NUM_CLASSES"] != \
           len(self.__dict__["CLASS_NAMES"]):
            raise ValueError("ERROR number of classses do not match, Classes: "
                             + str(self.__dict__["NUM_CLASSES"])
                             + " Class List: "
                             + str(self.__dict__["CLASS_NAMES"]))

    def print_contents(self):
        print(self.__dict__)

    def print_key_value(self):
        for key, value in self.__dict__.items():
            print(key, ":", value)


##############################################################


def load_file_names_Util(file_path,
                         image_ext,
                         full_file_path=True):
    """
      Returns a list of file names that can be just the file or fully qualified

      Args:
        file_path : path to the location of the file_exists
        image_est : the extension you want to filter on
        full_file_path : default True. True is full path, False is file name

      Returns:
        file_list : list of file names based on full_file_path
    """
    file_list = []
    file_names = os.listdir(file_path)
    for i, fn in enumerate(file_names):
        if fn.endswith(image_ext):
            if full_file_path:
                file_list.append(fn)
            else:
                head, tail = os.path.split(fn)
                file_list.append(tail)
    return file_list


##############################################################


def load_file_names_labeled_subdir_Util(file_path,
                                        file_ext,
                                        override_dirs=None,
                                        max_dir_files=1000000):
    """
      Get all subdirectories of basepath. Each represents a label.

      Args:
        file_path : path to the location of the file_exists
        image_est : the extension you want to filter on
        override_dirs : use only these directories to load files
        max_dir_files : maximum number of files to process.  This can be
            used for testing by setting the value smaller.

      Returns:
        file_list : list of file names based on full_file_path
        directories : list of directoies processed
    """
    file_list = []
    directories = [d for d in os.listdir(file_path)
                   if os.path.isdir(os.path.join(file_path, d))]
    if len(directories) == 0:
        print("Error with path, no subdirectories: ", file_path)
    else:
        if override_dirs is None:
            pass
        else:
            directories = override_dirs

        directories = sorted(directories)
        # Loop through the label directories and collect the data in a list
        for i, subdir in enumerate(directories):
            label_dir = os.path.join(file_path, subdir)
            print("loading subdir ", subdir, "  ", label_dir)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir)
                          if f.endswith(file_ext)]

            if len(file_names) > max_dir_files:
                file_names_random = sample(file_names, max_dir_files)
                print("Reducing files to ", str(max_dir_files), ": ",
                      subdir, " Actual ", len(file_names))
            else:
                file_names_random = file_names
                print("Adding ", subdir, len(file_names))

            for j, file_name in enumerate(file_names_random):
                # On a Mac, it sometimes includes ".DS_Store," do not include
                if not(".DS_Store" in file_name):
                    file_list.append(file_name)

    return file_list, directories


##############################################################

def string2image(string, shape=(96, 96)):
    """Converts a string of numbers to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape(shape)


##############################################################


def file_exists(file_id):
    # print(file_id, os.path.isfile(file_id))
    return os.path.isfile(file_id)


##############################################################


def save_list(list_name, file_name):
    """ Save a list to a file """
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(list_name)


##############################################################


def predictions_using_dataset(model_actual,
                              dataset,
                              steps,
                              batch_size,
                              create_bad_results_list=False):
    """
      Uses generator to predict results.  Builds actual_labels, predict_labels
      and predict_probabilities

      Args:
        model_actual : trained model to use for predictions
        ds_iter : dataset iterator
        steps : number of batches to process
        create_bad_results_list : bool default True.  Lets you trun on/off
            the creation of the bad results lists.

      Returns:
        actual_labels : list of actual labels
        predict_labels : list of predicted labels
        predict_probabilities : list of predicted probability array
        bad_results : list of bad results [actual_labels, predict_labels,
                      predict_probabilities, image]
    """

    bad_cnt = 0.0
    good_cnt = 0.0
    total_cnt = 0
    actual_labels = []
    predict_labels = []
    predict_probabilities = []
    bad_results = []

    for image_batch, label_batch in tqdm(dataset.take(steps)):
        for j in range(batch_size):
            image = image_batch[j]
            label = label_batch[j]

            total_cnt += 1
            # if a single label, then use it, otherwise find argmax()
            if label.shape[0] == 1:
                actual_label = label
            else:
                actual_label = np.argmax(label)

            image = np.expand_dims(image, axis=0)
            # image = tf.reshape(image, (1, *image.shape))

            predict_probabilities_tmp = model_actual.predict(image)[0]
            predict_label = np.argmax(predict_probabilities_tmp)

            actual_labels.append(actual_label)
            predict_labels.append(predict_label)
            predict_probabilities.append(predict_probabilities_tmp)

            correct_flag = actual_label == predict_label
            if correct_flag:
                good_cnt = good_cnt + 1
            else:
                bad_cnt = bad_cnt + 1
                if create_bad_results_list:
                    bad_results.append([[actual_label],
                                        [predict_label],
                                        predict_probabilities_tmp,
                                        np.squeeze(image)])
    print(" ")
    print("total: ", total_cnt, "  Good: ", good_cnt, "  Bad: ",
          bad_cnt, "  percent good: ", str(good_cnt/total_cnt))

    return actual_labels, predict_labels, predict_probabilities, \
        bad_results


##############################################################


def plot_results(arr1,
                 arr2,
                 labels,
                 x_label,
                 y_label,
                 title,
                 bar_labels,
                 num_labels):
    # Used by display_prediction_results to plot the results after a training
    # run
    plt.figure(figsize=(18, 4))
    index = np.arange(num_labels)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, arr1, bar_width,
                     alpha=opacity,
                     color='g',
                     label=bar_labels[0])

    rects2 = plt.bar(index + bar_width, arr2, bar_width,
                     alpha=opacity,
                     color='r',
                     label=bar_labels[1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if len(labels[0]) == 1:
        plt.xticks(index + bar_width, labels, fontsize=12)
    else:
        plt.xticks(index + bar_width, labels, rotation=90, fontsize=12)
    plt.legend()


def display_prediction_results(file_labels,
                               pred_labels,
                               pred_probabilities,
                               num_classes,
                               class_names_list):
    """
      Takes original and predicted results and graphs them.  Two graphs areas
      shown.  One for classes and one for predictions.

      Args:
        file_labels : original labels
        pred_labels : predicted labels
        pred_probabilities : predicted precentages
        num_classes : number of classes
        class_names_list : list of class names

      Returns:
        nothing
    """

    bad_cnt = 0.0
    good_cnt = 0.0
    total_cnt = len(pred_probabilities)
    bad_arr = np.zeros(num_classes)  # hold bad counts by label
    good_arr = np.zeros(num_classes)  # hold good counts by label
    bad_pred_arr = np.zeros(11)  # hold the segmented count, bad prodictions
    good_pred_arr = np.zeros(11)  # hold the segmented count, good prodictions

    for i, data in enumerate(pred_probabilities):

        actual_label = file_labels[i]
        predict_label = pred_labels[i]
        predict_probabilities = pred_probabilities[i]

        pred_seg = int(predict_probabilities[predict_label]*10)
        correct_flag = actual_label == predict_label
        if correct_flag:
            good_pred_arr[pred_seg] = good_pred_arr[pred_seg] + 1
            good_arr[predict_label] = good_arr[predict_label] + 1
            good_cnt = good_cnt + 1
        else:
            bad_pred_arr[pred_seg] = bad_pred_arr[pred_seg] + 1
            bad_arr[predict_label] = bad_arr[predict_label] + 1
            bad_cnt = bad_cnt + 1

    print("total: ", total_cnt, "  Good: ", good_cnt, "  Bad: ", bad_cnt,
          "  percent good: ", str(good_cnt/total_cnt))

    x_label = "Classes"
    y_label = "Count"
    title = "Prediction Counts by Class"
    bar_labels = ["Good", "Bad"]

    plot_results(good_arr, bad_arr, class_names_list, x_label, y_label,
                 title, bar_labels, num_classes)

    x_label = "Segments"
    y_label = "Count"
    title = "Predictions by Segments"
    bar_labels = ["Good", "Bad"]
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    plot_results(good_pred_arr, bad_pred_arr, label_names, x_label, y_label,
                 title, bar_labels, 11)


##############################################################


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Used by show_confusion_matrix.
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def show_confusion_matrix(labels,
                          predict_labels,
                          class_names,
                          show_graph=True):
    """
      Shows various accuracry measurements.

      Args:
        labels : actual labels
        predict_labels : predicted labels
        class_names : list of class names
        show_graph : flag to show or not show the actual graph.  set
                     to False for large number of classes.
      Returns:
        nothing
    """

    # Accuracy score
    print("Accuracy : " + str(accuracy_score(labels, predict_labels)))

    print("")

    # Classification report
    print("Classification Report")
    print(classification_report(np.array(labels),
                                np.array(predict_labels), digits=5))

    if show_graph:
        # Plot confusion matrix
        cnf_matrix = confusion_matrix(labels, predict_labels)
        print(cnf_matrix)
        plot_confusion_matrix(cnf_matrix, classes=class_names)


##############################################################
# Displays the Training and Validation files in a montage format


def batch_montage_display_using_generator(image_batch, img_channels):
    # images already read into 4 dim arrays as batches
    """
      Uses generator to display a montage.  Useful to check output of a
      generator.

      Args:
        image_batch : batch of images
        img_channels : number of channels

      Returns:
        nothing
    """

    multi_ch = True
    if img_channels == 1:
        image_batch = np.reshape(image_batch,
                                 (image_batch.shape[0],
                                  image_batch.shape[1],
                                  image_batch.shape[2]))
        multi_ch = False

    img_batch_montage = montage(image_batch, multichannel=multi_ch)
    fig, (ax1) = plt.subplots(1, 1, figsize=(30, 10))
    if img_channels == 1:
        ax1.imshow(img_batch_montage, cmap="gray")
    else:
        ax1.imshow(img_batch_montage)
    ax1.set_title('Batch images: '+str(len(image_batch)))
    # Uncomment if you want to save the figure
    # fig.savefig('overview.png')


##############################################################


def image_show_seq_model_layers_BETA(image_path,
                                     model,
                                     image_dim,
                                     layers_num=4,
                                     activation_layer_num=0,
                                     activation_channel_num=9):
    """
      BETA, works but not fully tested.  Displays the activation layers_num
      for a model.  Model can be fully tained or just initial weights.
      This is a merging of these two articles:
        Orig: https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part1.ipynb#scrollTo=-5tES8rXFjux
        Another example https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md

      Args:
        image_path : fully qualified path to the image to be used
        model : model to use, can be trainned or just initialized
        image_dim : dimesion of the image (r, c, d)
        layers_num : number of layers to show, should be > 2
        activation_layer_num : number of the activation layer
        activation_channel_num : number of the channel
        (layer and channel are used to display the image)

      Returns:
        nothing
    """

    # Let's define a new Model that will take an image as input, and will
    # output intermediate representations for all layers in the previous model
    # after the first.
    successive_outputs = [layer.output for layer in model.layers[:layers_num]]
    visualization_model = Model(model.input, successive_outputs)

    if image_dim[2] == 1:
        img = load_img(image_path,
                       color_mode="grayscale",
                       target_size=(image_dim[0], image_dim[1]))
        cmap = "gray"
        plt.matshow(img, cmap=cmap)
    else:
        img = load_img(image_path, target_size=image_dim)
        cmap = "viridis"
        plt.matshow(img)

    print("Loaded image: ", image_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # Plots the "activation_channel" of the "activation_layer"
    layer_activation = successive_feature_maps[activation_layer_num]
    print("Layer: ", activation_layer_num,
          "  Channel: ", activation_channel_num,
          "  Shape: ", layer_activation.shape)
    plt.matshow(layer_activation[0, :, :, activation_channel_num], cmap=cmap)

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the
            # fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size:(i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap=cmap)

##############################################################
