import os
import cv2
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='images to video')
    parser.add_argument('--sim_img_dir', type=str)
    parser.add_argument('--sensor_img_dir', type=str)
    parser.add_argument('--output_path', type=str, default='output.avi')
    args = parser.parse_args()

    sim_img_dir = args.sim_img_dir
    sim_img_list = os.listdir(sim_img_dir)
    sim_img_list.sort()
    sensor_img_dir = args.sensor_img_dir
    sensor_img_list = os.listdir(sensor_img_dir)
    sensor_img_list.sort()
    
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (400, 600))

    for i in range(len(sim_img_list)):
        sim_img = cv2.imread(os.path.join(sim_img_dir, sim_img_list[i]))
        sim_img = sim_img[150:-150, 300:-300]
        sim_img = cv2.resize(sim_img, (400, 300), interpolation=cv2.INTER_AREA)
        sensor_img = cv2.imread(os.path.join(sensor_img_dir, sensor_img_list[i]))
        sensor_img = cv2.resize(sensor_img, (400, 300), interpolation=cv2.INTER_AREA)
        img = np.concatenate([sim_img, sensor_img], axis=0)
        out.write(img)

    out.release()

if __name__ == '__main__':
    main()