import os
import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, isGray = False):
    # load image
    img = skimage.io.imread(path, as_gray = isGray)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def show_image(image):
    skimage.io.imshow(image * 255)
    skimage.io.show()
    return

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)

def indexTraininData(path):
    color = []
    depth = []
    for r, _, f in os.walk(path):
        for file in f:
            if 'StereoL' in file:
                color.append(os.path.join(r, file))
            elif 'DepthCamera' in file:
                depth.append(os.path.join(r, file))
    assert len(color) > 0 and len(depth) > 0
    color.sort()
    depth.sort()
    return color, depth

def prepareData(path):
    color, depth = indexTraininData(path)
    assert len(color) == len(depth)
    dataSet = [None] * len(color)
    for i in range(len(color)):
        dataSet[i] = (color[i], depth[i])
    return dataSet
    
    
if __name__ == "__main__":
    test()
