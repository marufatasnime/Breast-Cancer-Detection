import cv2 as vision
import pandas as panda
import os
from directories import SOURCE_TRAIN_DIR, SOURCE_TEST_DIR, TRAIN_DIR, TEST_DIR, TRAIN_LABELS_DIR, TEST_LABELS_DIR


class SortImages:
    train_labels_file = panda.read_csv(TRAIN_LABELS_DIR)
    test_labels_file = panda.read_csv(TEST_LABELS_DIR)

    train_labels = train_labels_file['class'].to_list()
    test_labels = test_labels_file['class'].to_list()

    def apply(self):
        self.sort_images(self.train_labels, SOURCE_TRAIN_DIR, TRAIN_DIR)
        self.sort_images(self.test_labels, SOURCE_TEST_DIR, TEST_DIR)

    def sort_images(self, labels, source, destination):
        index = 0
        for file_name in os.listdir(source):
            image = vision.imread(os.path.join(source, file_name), 0)
            vision.imwrite(os.path.join(f'{destination}/{labels[index]}', file_name), image)
            index += 1
        print(f'Sorted {index} images')


class ChangeImageFormat:
    def __init__(self, new_format):
        self.format = new_format

    def apply(self):
        self.change_format(SOURCE_TRAIN_DIR)
        self.change_format(SOURCE_TEST_DIR)

    def change_format(self, directory):
        counter = 0
        for file_name in os.listdir(directory):
            image_file = os.path.join(directory, file_name)
            image_file_name = image_file.split('.')
            if image_file_name[-1] != self.format:
                image = vision.imread(image_file, 0)
                vision.imwrite(image_file_name[0] + '.' + self.format, image)
                counter += 1
        print(f'Converted {counter} images to {self.format}')


if __name__ == '__main__':
    sort_images = SortImages()
    sort_images.apply()
