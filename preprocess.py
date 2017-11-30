import os
from twisted.logger.test import test_file

import numpy as np
import cv2
import csv
import random
from sklearn.model_selection import train_test_split


def remove_txt(dirname):
    '''
    :param dirname: name of the directory
    :return:  walks through directory and removes texts
    '''
    for cur, _dirs, files in os.walk(dirname):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)
        for f in files:
            if ".txt" in f:
                os.remove("data/faces/" + tail + "/" + f)


def find_class(param, file_name):
    '''
    :param param: folder where the data is in
    :param file_name: the name of the image
    :return: number that represents a class that image represents given file name and csv that contains the data
    '''
    for cur, _dirs, files in os.walk(param):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)
        for f in files:
            if ".csv" in f:
                file_path = param + "/" + f
                train_data = list(csv.reader(open(file_path), delimiter='\t'))
                for i in range(len(train_data)):
                    image_name = train_data[i][2] + "." + train_data[i][1]
                    if image_name in file_name:
                        print train_data[i][3]
                        class_label = get_classification(train_data[i][3])
                        if class_label is None:
                            return 0
                        else:
                            return class_label


def get_classification(input):
    '''
    :param input: tuple or string representing age
    :return: number representing one of the age classes
    '''
    print "input ", input
    if input is None:
        return 0
    class_num = 0
    num = eval(input)
    if isinstance(num, tuple):
        class_num = (num[0] + num[1]) / 2
    elif isinstance(num, int):
        class_num = num

    if 0 < class_num < 10:
        return 1
    elif 10 <= class_num < 20:
        return 2
    elif 20 <= class_num < 30:
        return 3
    elif 30 <= class_num < 40:
        return 4
    elif 40 <= class_num < 55:
        return 5
    elif 55 <= class_num < 65:
        return 6
    elif 65 <= class_num < 90:
        return 7
    else:
        return 0


def generate_train_csv(f):
    '''
    :param f: file path
    :return:
    '''
    count = 0
    class_labels = []
    class_labels_num = np.zeros(9)

    my_file = open("data/data.csv", 'wb')
    writer2 = csv.writer(my_file)
    for cur, _dirs, files in os.walk(f):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)
        for f in files:
            if ".jpg" in f:
                path = "data/faces/" + tail + "/" + f
                class_num = find_class("data/class_label", f)
                string_written = [path, f, class_num]
                writer2.writerow(string_written)
                my_file.flush()
                print count, string_written, f
                count += 1
                if class_num not in class_labels:
                    class_labels.append(class_num)
                class_labels_num[class_labels.index(class_num)] += 1
    my_file.close()

    for i in range(len(class_labels)):
        print class_labels[i], "-->", class_labels_num[i]


def make_train_test(main_file, train_file, test_file, test_size, random_state):
    '''
    :param data_file: file containing full data thats waiting to be shuffeled and broken to train and test
    :param train_file: training file to be created
    :param test_file:  testing file to be created
    :param test_size: % of test data
    :param random_state: randomizing input
    :return:
    '''
    data = list(csv.reader(open(main_file)))
    random.shuffle(data)
    random.shuffle(data)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    random.shuffle(train)
    random.shuffle(test)

    X = np.asarray(data)
    Y = np.asarray(train)
    z = np.asarray(test)
    print "prev shape : ", X.shape, "train size ", Y.shape, ":test size", z.shape

    my_file = open(train_file, 'wb')
    writer2 = csv.writer(my_file)
    for i in range(len(train)):
        writer2.writerow(train[i])
    my_file.flush()
    my_file.close()

    my_file = open(test_file, 'wb')
    writer2 = csv.writer(my_file)
    for i in range(len(test)):
        writer2.writerow(test[i])
    my_file.flush()
    my_file.close()


def make_train_test_folders(train_file, test_file):
    data = list(csv.reader(open(test_file)))
    for i in range(len(data)):
        image = cv2.imread(data[i][0])
        name = "data/images/test/" + str(data[i][2]) + "/" + data[i][1]
        print name
        cv2.imwrite(name, image)


# remove_txt("data/")
# generate_train_csv("data/")
# make_train_test("data/data.csv", "data/data_train2.csv", "data/data_test2.csv", test_size=0.2, random_state=40)
# make_train_test_folders("data/data_train2.csv", "data/data_test2.csv")
