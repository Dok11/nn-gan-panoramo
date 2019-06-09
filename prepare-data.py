import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array

SIZE_X = 128
SIZE_Y = 64
CURRENT_DIR: str = os.getcwd()
DATA_SLICE: int = 1


def get_data():
    result = []

    dir_root = os.path.join(CURRENT_DIR, 'train-images')
    file_list = os.listdir(os.path.join(dir_root, dir_root))

    for file_name in file_list:
        result.append(os.path.join(dir_root, file_name))

    return result


def set_data():
    data = get_data()

    for slice_index in range(DATA_SLICE):
        print('start work slice with', slice_index, 'of', DATA_SLICE)
        data_list = []

        for image_name in data[slice_index::DATA_SLICE]:
            image_loaded = load_img(image_name,
                                    color_mode='grayscale',
                                    target_size=(SIZE_Y, SIZE_X),
                                    interpolation='bicubic')
            image = img_to_array(image_loaded)
            data_list.append(image)

        slice_str = str(slice_index).zfill(3)
        train_file = os.path.join(CURRENT_DIR, 'data', 'train' + slice_str)
        train_data = np.array(data_list).astype('uint8')
        np.savez(train_file, train_data=train_data)


set_data()
