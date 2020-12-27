from torch import device as device_
from facenet_pytorch import MTCNN
from torch import cuda
from tqdm import tqdm
import numpy as np
import cv2


device = device_('cuda:0' if cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=200, device=device, selection_method="probability")


def extract_face_from_image(image: np.array):
    # Image.fromarray(image).show()
    res = mtcnn.detect(image)
    if res[0] is None: return None
    box = res[0][0].astype(int)
    y1, y2, x1, x2 = box[1], box[3], box[0], box[2]
    # Image.fromarray(image[y1:y2, x1:x2]).show()
    return image[y1:y2, x1:x2]

def extract_frame_from_video(path: str, frame_idx_start=1, frame_idx_end=None, step=1):
    faces = {}
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx_end is not None and frame_idx_end > length: length = frame_idx_end
    print(f"Количество кадров: {length}\t ФПС: {fps:.2f}")

    for i in tqdm(range(1, length + 1)):
        ret, img = cap.read()
        # show_image(img)
        # return
        if not ret:
            break
        if i >= frame_idx_start:
            if int(i - 1) % step == 0:
                face = extract_face_from_image(img)
                if face is None: print(i, 'None')
                faces[i] = {'FullFace': face, 'Time': (1 / fps) * (i - 1)}
    fps /= step
    print(f"Новое количество кадров: {len(faces.keys())}\t Новый ФПС: {fps:.2f}")
    return faces, fps, length
