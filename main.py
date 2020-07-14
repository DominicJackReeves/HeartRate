import sys
import numpy as np
import subprocess
import argparse
import cv2
import face_detection
import fourier_transform

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="The path to your required video.")
    ap.add_argument("-f", "--sample", required=True,
        help="The frequency you want to sample the video at. Eg. -f 8 will mean you sample 8 frames a second. 0 will mean you stay at the base sample rate.")
    ap.add_argument("-s", "--span", required=True,
        help="The time you want to average heart rate over. Higher values mean less responsive to change but an overall more accurate value.")
    args = vars(ap.parse_args())

    
    average_list = []
    cap = cv2.VideoCapture(args["video"])

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)*0.97
    sample = args["sample"]
    if int(sample) == 0:
        sample = fps
    frameList = np.arange(0,frames,np.ceil(int(fps)//int(sample)))
    for frame in frameList:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        print(frame)
        ret, image = cap.read()
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
        s2_out = face_detection.initiate(image)
        average_list.append(s2_out)
        print(average_list[-1])
    
    np.save("facial_average.npy", average_list)
    # fourier_transform.heartbeat(fps, int(sample), args["span"])

