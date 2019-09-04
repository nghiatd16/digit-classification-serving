import numpy as np
import cv2
import os

def _read_object_like(object_like, read_fn, *read_fn_args, **read_fn_kwargs):
    '''
    Hàm để đọc các object linh động, nếu input là đường dẫn thì sẽ tiến hành đọc file, hoặc url.
    Nếu input đã là một object cần xử lý thì chỉ đơn giản return về.
    Args:
        - object_like(str hoặc object): string thì chứa đường dẫn file, object thì chỉ return về chính input
        - read_fn (function): hàm xử lý để đọc nội dung nếu object_like là đường dẫn hay url.
        - *read_fn_args (parameter list): args truyền vào read_fn
        - **kwargs (parameter dictionary): kwargs truyền vào read_fn
    Return: (object) - Object cần đọc từ đường dẫn, nếu input đã là object, nhận về input
    '''
    if object_like.__class__ is str:
        if not os.path.isfile(object_like):
            raise FileNotFoundError()
        return read_fn(object_like, *read_fn_args, **read_fn_kwargs)
    return object_like

def preprocess_bgr(image_like):
    img = _read_object_like(image_like, cv2.imread)
    assert img.ndim == 3 and img.shape[-1] >= 3, "Function requires 3 channels image"
    img = img[:,:,:3]
    if img.shape[:2] != (28,28):
        img = cv2.resize(img, (28,28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    img = img/255.0
    return img

def _base_preprocess_gray(image_like):
    img = _read_object_like(image_like, cv2.imread, 0)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if img.shape[:2] != (28,28):
        img = cv2.resize(img, (28,28))
    img = np.expand_dims(img, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    img = img.astype(np.uint8)
    return img

def preprocess_gray_nobg(image_like):
    img = _base_preprocess_gray(image_like)
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = img/255.0
    img = np.reshape(img, (28,28,1))
    return img


def preprocess_gray_bg(image_like):
    img = _base_preprocess_gray(image_like)
    img = img/255.0
    img = np.reshape(img, (28,28,1))
    return img

def preprocess_gray_images(image_digits_like):
    # print(len(image_digits_like))
    image_digits = [_base_preprocess_gray(image_like) for image_like in image_digits_like]
    
    # Process gray image with background
    gray_bg_images = image_digits.copy()
    gray_bg_images = np.stack(gray_bg_images)/255.0
    gray_bg_images = np.reshape(gray_bg_images, (-1,28,28,1))

    # Process gray image with no background
    gray_nobg_images = [cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for img in image_digits]
    gray_nobg_images = np.stack(gray_nobg_images)/255.0
    gray_nobg_images = np.reshape(gray_nobg_images, (-1,28,28,1))

    return gray_bg_images, gray_nobg_images

def preprocess_unet(image_like):
    img = _read_object_like(image_like, cv2.imread)
    img = img[:,:,:3]
    img = cv2.resize(img, (224, 64))
    img = img/255.0
    return img