import numpy as np
import cv2



def fill_None(faces_dict: dict):
    filled = 0
    prev = None
    next_ = None
    keys = [key for key in faces_dict]

    for key in keys:
        if faces_dict[key]['FullFace'] is None:
            if prev is not None:
                filled += 1
                faces_dict[key]['FullFace'] = prev.copy()
        else:
            prev = faces_dict[key]['FullFace']

    keys.reverse()
    for key in keys:
        if faces_dict[key]['FullFace'] is None:
            if next_ is not None:
                filled += 1
                faces_dict[key]['FullFace'] = next_.copy()
        else:
            next_ = faces_dict[key]['FullFace']

    if prev is None and next_ is None:
        raise "Нет ни единого кадра с лицом!!!"

    print("Количество пропусков -", filled)

    pass


def resize_images_to_one_shape(faces_dict: dict, shape=None):
    if shape is None:
        shapes = []
        for key in faces_dict:
            shapes += [faces_dict[key]['FullFace'].shape[:2]]
        shape = np.mean(shapes, axis=0).astype(int)
    for key in faces_dict:
        faces_dict[key]['FullFace'] = cv2.resize(faces_dict[key]['FullFace'], (shape[1], shape[0]))
    pass
