import cv2
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures


url = 'http://172.20.10.12'

cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
while True:
        img_resp = urllib.request.urlopen(url)