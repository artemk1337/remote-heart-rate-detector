from PIL import Image
import cv2


def create_ROI_using_size(image, height, width, left_shift=0.1, right_shift=0.1,
                          top_shift=0.1, bottom_shift=0.1, rectangled_image=None, show=False):
    shift_right_left = (int(left_shift * width), int(width - right_shift * width))
    shift_top_bottom = (int(top_shift * height), int(height - bottom_shift * height))
    if rectangled_image is None: rectangled_image = image.copy()
    rectangled_image = cv2.rectangle(rectangled_image,
                                    (shift_right_left[0], shift_top_bottom[0]),
                                    (shift_right_left[1], shift_top_bottom[1]),
                                    (255, 0, 0), 5)
    if show: Image.fromarray(rectangled).show()
    return image[shift_top_bottom[0]:shift_top_bottom[1], shift_right_left[0]:shift_right_left[1]], rectangled_image

def create_ROI(faces_dict: dict):
    for key in faces_dict.keys():
        fullface = faces_dict[key]['FullFace']
        # print(fullface.shape)
        left_cheek, _ = create_ROI_using_size(fullface, *fullface.shape[:2], 0.15, 0.7, 0.5, 0.4)
        right_cheek, _ = create_ROI_using_size(fullface, *fullface.shape[:2], 0.7, 0.15, 0.5, 0.4, _)
        forehead, _ = create_ROI_using_size(fullface, *fullface.shape[:2], 0.3, 0.3, 0.1, 0.8, _)
        faces_dict[key]['Forehead'] = forehead
        faces_dict[key]['LeftCheek'] = left_cheek
        faces_dict[key]['RightCheek'] = right_cheek
        faces_dict[key]['Rectangled'] = _
