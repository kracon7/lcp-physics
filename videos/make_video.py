import os
import cv2
import numpy as np

frameSize = (424, 240)

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')
img_list = os.listdir(img_dir)
img_list.sort()
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, frameSize)
# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, frameSize)

# st, ed = 121, 179
# for i in range(st, ed):
#     filename = os.path.join(img_dir, img_list[i])
#     img = cv2.imread(filename)
#     img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
#     out.write(img)

for i in range(len(img_list)):
    filename = os.path.join(img_dir, img_list[i])
    img = cv2.imread(filename)
    img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
    out.write(img)

out.release()