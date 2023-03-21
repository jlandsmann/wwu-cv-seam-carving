import csv
import numpy as np
import random

PATH_TO_METADATA_CSV = "../gecarvteBilder/metadaten.csv"
PATH_TO_CARVED_CSV = "../gecarvteBilder/carved_labels.csv"
PATH_TO_NON_CARVED_CSV = "../gecarvteBilder/non_carved_labels.csv"

PATH_TO_TEST_LABELS_CSV = "../gecarvteBilder/test_labels.csv"
PATH_TO_TRAIN_LABELS_CSV = "../gecarvteBilder/train_labels.csv"



def split_all_data_in_class_folders():

    carved_csv = []
    non_carved_csv = []

    # go through all files
    with open(PATH_TO_METADATA_CSV) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if row[2] == '1' and row[3] != '1':
                carved_csv.append(row)
            else:
                row[2] = '0'
                non_carved_csv.append(row)

        print(len(carved_csv))
        print(len(non_carved_csv))

    arr = np.asarray(carved_csv)
    np.savetxt(PATH_TO_CARVED_CSV, arr, fmt='%s', delimiter=';')

    arr = np.asarray(non_carved_csv)
    np.savetxt(PATH_TO_NON_CARVED_CSV, arr, fmt='%s', delimiter=';')


def create_csvs_for_test_and_train(test_train_ratio, carved_non_carved_ratio):
    carved_csv = []
    non_carved_csv = []

    train_csv = []
    test_csv = []



    # go through all files
    with open(PATH_TO_METADATA_CSV) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for row in reader:
            if row[2] == '1' or row[3] == '1':
                row[2] = '1'
                carved_csv.append(row)
            else:
                non_carved_csv.append(row)

    full_count_carved = len(non_carved_csv)
    full_count_non_carved = len(non_carved_csv)

    if carved_non_carved_ratio <= len(carved_csv)/len(non_carved_csv):
        full_count_carved = len(non_carved_csv)*carved_non_carved_ratio
        full_count_non_carved = len(non_carved_csv)
    else:
        full_count_carved = len(carved_csv)
        full_count_non_carved = len(carved_csv)/carved_non_carved_ratio

    carved_in_train = int(full_count_carved * test_train_ratio)
    carved_in_test = full_count_non_carved - carved_in_train

    non_carved_in_train = int(full_count_non_carved * test_train_ratio)
    non_carved_in_test = full_count_non_carved - non_carved_in_train

    carved_index = 0
    non_carved_index = 0

    for i in range(carved_in_train):
        train_csv.append(carved_csv[carved_index])
        carved_index += 1

    for i in range(carved_in_test):
        test_csv.append(carved_csv[carved_index])
        carved_index += 1

    for i in range(non_carved_in_train):
        train_csv.append(non_carved_csv[non_carved_index])
        non_carved_index += 1

    for i in range(non_carved_in_test):
        test_csv.append(non_carved_csv[non_carved_index])
        non_carved_index += 1

    random.shuffle(train_csv)
    random.shuffle(test_csv)

    arr = np.asarray(train_csv)
    np.savetxt(PATH_TO_TRAIN_LABELS_CSV, arr, fmt='%s', delimiter=';')

    arr = np.asarray(test_csv)
    np.savetxt(PATH_TO_TEST_LABELS_CSV, arr, fmt='%s', delimiter=';')

    print(carved_in_train)
    print(carved_in_test)
    print(non_carved_in_train)
    print(non_carved_in_test)




split_all_data_in_class_folders()

#create_csvs_for_test_and_train(test_train_ratio=0.8, carved_non_carved_ratio=1)
