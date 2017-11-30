import sys
import argparse
import numpy as np
from PIL import Image
from facecrop import facecrop
import os
import cv2

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

target_size = (100, 100)  # fixed size for InceptionV3 architecture


def predict(model, img, target_size):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
    Returns:
      list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def convert_graysale(f):
    '''
    :param f: file name
    :return: returns the grayscaled and scaled version of the image
    '''
    image = cv2.imread(f)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    resized_image = cv2.resize(gray_image, target_size)
    cv2.imwrite(f, resized_image)


def get_classNames(input):
    '''
    :param input: tuple or string representing age
    :return: number representing one of the age classes
    '''

    if input == 0:
        return "between 0 and 10"
    elif input == 1:
        return "between 10 and 20"
    elif input == 2:
        return "between 20 and 30"
    elif input == 3:
        return "between 30 and 40"
    elif input == 4:
        return "between 40 and 55"
    elif input == 5:
        return "between 55 and 65"
    elif input == 6:
        return "above 65"
    else:
        return "unknown"


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    a.add_argument("--image_array")
    a.add_argument("--model")
    args = a.parse_args()

    if args.image is None:
        a.print_help()
        sys.exit(1)

    model = load_model(args.model)
    if args.image is not None:
        facecrop(args.image)
        fname, ext = os.path.splitext(args.image)
        # convert_graysale(fname + "_cropped_" + ext)
        img = Image.open(fname + "_cropped_" + ext)
        preds = predict(model, img, target_size)
        class_num = np.argmax(preds)
        print "\n"
        print  get_classNames(class_num)
        os.remove(fname + "_cropped_" + ext)
