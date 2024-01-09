import numpy as np


def create_square_images(img_lst, ncols):
    img_array = []
    img_row_array = []
    for curr_img in img_lst:
        if len(img_row_array) < ncols - 1:
            img_row_array.append(curr_img)
        else: # len(img_row_array) == ncols
            assert len(img_row_array) == ncols - 1
            img_row_array.append(curr_img)
            img_array.append(np.concatenate(img_row_array, axis=1))
            img_row_array = []
    return np.concatenate(img_array, axis=0) 


# TODO: include more image augmentation methods