import os
import cv2
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='images to video')
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--output_path', type=str, default='output.avi')
    args = parser.parse_args()

    frameSize = (424, 240)

    # img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')
    img_dir = args.img_dir
    img_list = os.listdir(img_dir)
    img_list.sort()
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, frameSize)

    for i in range(len(img_list)):
        filename = os.path.join(img_dir, img_list[i])
        img = cv2.imread(filename)
        img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
        out.write(img)

    out.release()

if __name__ == '__main__':
    main()