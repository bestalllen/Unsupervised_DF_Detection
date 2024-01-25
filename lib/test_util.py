import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import restoration
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
from scipy.stats import spearmanr
import statistics
import os

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def get_crop(frame, mtcnn):
    height, width = frame.shape[:2]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        continue
    x, y, size = get_boundingbox(boxes[0].flatten(), width, height)
    cropped_face = frame[y:y + size, x:x + size]
    cropped_face = cv2.resize(cropped_face, (299, 299))
    cropped_face = Image.fromarray(cropped_face)

    return cropped_face