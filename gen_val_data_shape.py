import cv2
import os

new_val_data = 'data/voc0712/2007_test_wh.txt'
new_val = open(new_val_data, 'w')

val_data = 'data/voc0712/2007_test.txt'
with open(val_data, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i%100 == 0:
            print('processing:', i, '/', len(lines))
        img_path = line.strip().split()[0]
        img = cv2.imread(img_path)
        if img is None:
            print('can not load image: ', img_path)
            continue
        new_val.write(img_path + ' ' + str(img.shape[1]) + ' ' + str(img.shape[0]) + '\n')
new_val.close()