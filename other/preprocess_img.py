import PIL
import numpy as np

MEAN_VALUES = np.array([104, 117, 123])
IMAGE_W = 224


def preprocess(img):
    # convert RGB to BGR
    img = img[:, :, [2, 1, 0]]
    # sum mean
    img = img - MEAN_VALUES
    # convert from [w,h,3 to 1,3,w,h]
    img = np.transpose(img, (2, 0, 1))[None]
    return img


def deprocess(img):
    img = img.reshape(img.shape[1:]).transpose((1, 2, 0))
    for i in range(3):
        img[:, :, i] += MEAN_VALUES[i]
    return img[:, :, :: -1].astype(np.uint8)


def resize_img(img_np, base_s=IMAGE_W):
    img = PIL.Image.fromarray(np.uint8(img_np))
    w, h = img.size
    if w < h:
        h = base_s * h // w
        w = base_s
    else:
        w = base_s * w // h
        h = base_s
    img = img.resize((w, h), PIL.Image.ANTIALIAS)
    y, x = h // 2, w // 2
    img = img.crop((x - base_s // 2, y - base_s // 2, x + base_s // 2, y + base_s // 2))

    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)
        img_np = np.concatenate((img_np,) * 3, axis=2)

    return img_np
