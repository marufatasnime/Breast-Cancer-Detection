import cv2 as vision
import pandas as panda
import os
from time import time
from numpy import array, ones, hstack, vstack, uint8
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


class ImageProcessor:
    processed_image = 0

    def generate_mask(self, image, name):
        self.processed_image += 1
        start_time = time()
        print('Processing Image: ', name)
        binary_mask = []
        for rows in image:
            values = []
            for pixel in rows:
                if pixel > 10:
                    values.append(255)
                else:
                    values.append(0)
            binary_mask.append(values)

        binary_mask = array(binary_mask, dtype=uint8)

        struct_element = ones((3, 3), uint8)
        closed_mask = vision.morphologyEx(
            binary_mask,
            vision.MORPH_CLOSE,
            struct_element
        )

        struct_element = ones((120, 120), uint8)
        mask = vision.morphologyEx(
            closed_mask,
            vision.MORPH_OPEN,
            struct_element
        )

        cleaned_image = vision.bitwise_and(image, mask)
        comparison_image = vstack(
            (hstack((binary_mask, closed_mask, mask)),
             hstack((image, cleaned_image, mask)))
        )
        vision.imwrite(f'resources/exports/cmp_{name}', comparison_image)
        end_time = time()
        return end_time - start_time

    def clean_images(self):
        start_time = time()
        for file_name in os.listdir(SOURCE_TRAIN_DIR):
            image_file = os.path.join(SOURCE_TRAIN_DIR, file_name)
            image = vision.imread(image_file, 0)
            time_taken = self.generate_mask(image, file_name)
            print(f'Time taken: {round(time_taken, 2)} seconds')
            print(f'Processed {self.processed_image} images \n')
        print('Process Finished')

        for file_name in os.listdir(SOURCE_TEST_DIR):
            image_file = os.path.join(SOURCE_TEST_DIR, file_name)
            image = vision.imread(image_file, 0)
            time_taken = self.generate_mask(image, file_name)
            print(f'Time taken: {round(time_taken, 2)} seconds')
            print(f'Processed {self.processed_image} images \n')
        end_time = time()
        print(f'Process Finished in {end_time - start_time} seconds.')

    def extract_roi(self):
        pass

    def segment_image(self):
        pass

    def show_image(self, image=None, compare=False, name=None):
        if image is None:
            image = self.image
        if compare:
            image = hstack((self.image, image))
        if name is not None:
            vision.imwrite(name, image)
        vision.imshow('image', image)
        key = vision.waitKey(0)
        if key == 27:
            vision.destroyAllWindows()


if __name__ == '__main__':
    print("Preprocess Images\n")
    image_processor = ImageProcessor()
    image_processor.clean_images()
