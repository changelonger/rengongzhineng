import cv2
import argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--image", type = str, help = "image path")
args = ap.parse_args()

image = cv2.imread(args.image)