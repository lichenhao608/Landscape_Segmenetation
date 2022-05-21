import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt

true_image = plt.imread('data/Training_Color.jpg').copy()
class_image = plt.imread('data/ChbarMon_LandClasses.tif').copy()

print(true_image.dtype)
print(class_image.dtype)

for i, val in enumerate(np.unique(class_image)):
    class_image[class_image == val] = i
h, w = class_image.shape

train_i = 0
test_i = 0

training = []
testing = []
for size in range(100, 150):
    starth = np.random.randint(0, 50)
    startw = np.random.randint(0, 50)
    steph = 1 if h - size - starth <= 10 else(h - size - starth) // 10
    stepw = 1 if w - size - startw <= 10 else(w - size - startw) // 10
    for col in range(startw, w - size, stepw):
        for row in range(starth, h - size, steph):
            imx = true_image[row:row + size, col:col + size]
            imy = class_image[row:row + size, col:col + size]

            if np.random.random() < 0.9:
                img_path = 'data/train/images/train_{:05d}.jpg'.format(train_i)
                seg_path = 'data/train/annotations/train_{:05d}.png'.format(train_i)
                Image.fromarray(imx).save(img_path)
                Image.fromarray(imy).save(seg_path)
                train_i += 1
                training.append({
                    'fpath_img': img_path,
                    'fpath_segm': seg_path,
                    'width': size, 'height': size
                })
            else:
                img_path = 'data/valid/images/test_{:05d}.jpg'.format(test_i)
                seg_path = 'data/valid/annotations/test_{:05d}.png'.format(test_i)
                Image.fromarray(imx).save(img_path)
                Image.fromarray(imy).save(seg_path)
                test_i += 1
                testing.append({
                    'fpath_img': img_path,
                    'fpath_segm': seg_path,
                    'width': size, 'height': size
                })

with open('data/training.odgt', 'w') as f:
    for k in training:
        json.dump(k, f)
        f.write('\n')

with open('data/testing.odgt', 'w') as f:
    for k in testing:
        json.dump(k, f)
        f.write('\n')
 
