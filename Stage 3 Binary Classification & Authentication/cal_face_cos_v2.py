# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from scipy.spatial import distance
import glob
import random
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
import torch
from model import SupCEXceptionNet
from scipy.stats import spearmanr
import os
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
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

# 加载人脸检测器和人脸关键点检测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/home/xsc/experiment/LRNet-main/shape_predictor_68_face_landmarks.dat")

# 计算两个向量之间的余弦相似度
# def cosine_similarity(a, b):
#     return 1 - distance.cosine(a, b)

# def calculate_cosine_similarity(feature_a, feature_b):
#     # 计算余弦相似度
#     similarity = cosine_similarity([feature_a], [feature_b])[0][0]
#     return similarity
from sklearn.metrics.pairwise import cosine_similarity
def calculate_cosine_similarity(feature_a, feature_b):
    # 计算余弦相似度
    similarity = cosine_similarity([feature_a], [feature_b])[0][0]
    return similarity

# 计算人脸特征向量
def compute_face_descriptor(image, shape):
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# 加载人脸识别模型
# face_rec_model = dlib.face_recognition_model_v1("/home/xsc/experiment/Exploiting-Visual-Artifacts-master/dlib_face_recognition_resnet_model_v1.dat")

# 加载视频
video_folder_1 = glob.glob('/dataset/DFD/select_100fake/*')
video_folder_1 =  random.sample(video_folder_1, 100)
video_folder_0 = glob.glob('/dataset/DFD/select_100real/*')
video_folder_0 =  random.sample(video_folder_0, 100)
video_folder = video_folder_1 + video_folder_0
similarity_total_1 = []
similarity_total_0 = []

test_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))])
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

filename = '/home/20_xsc/experiments/Inter-frame Cal/ours_dirty_rate10%.pth'
model = SupCEXceptionNet()
# backbone_model.fc = nn.Identity()
# model = torch.nn.DataParallel(model)
checkpoint = torch.load(filename, map_location=device)
state_dict = checkpoint['model']
model = model.cuda()
print(model.load_state_dict(state_dict))

mtcnn = MTCNN(device='cuda:1').eval()

for i in video_folder:
    print(i)
    cap = cv2.VideoCapture(i)

    # 存储前一帧的人脸特征向量
    prev_face_descriptor = None

    similarity_sum = []
    num = 0
    # 读取视频帧并进行人脸识别
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
        img = test_transform(cropped_face)
        # 调整输入的形状为[1, channels, height, width]
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = img.cuda(non_blocking=True)
        output,_ = model(img)
        output = output.cpu()
        output = output.detach().numpy()
        
        similarity = 0
        # 在这里进行人脸余弦相似度的计算，与前一帧人脸特征向量进行比较
        
        if prev_face_descriptor is not None:
            similarity, p_value = spearmanr(output[0], prev_face_descriptor[0])
            # print(output[0])
            # print("--------")
            # print(prev_face_descriptor[0])
            # similarity = calculate_cosine_similarity(output, prev_face_descriptor)

        # 更新前一帧的人脸
        prev_face_descriptor = output
        print(similarity)
        similarity_sum.append(similarity) 
        num = num + 1
        if num>=32:
            break
    # 剔除0值
    filtered_data = [x for x in similarity_sum if x != 0]
    similarity_sum = filtered_data
    total = sum(similarity_sum)
    similarity_avg = total / len(similarity_sum)
    if i.split('/')[-2]=='select_100real':
        similarity_total_0.append(similarity_avg)
    elif i.split('/')[-2]=='select_100fake':
        similarity_total_1.append(similarity_avg)

similarity_total = similarity_total_0 + similarity_total_1
print(len(similarity_total))

# 打开文件，以写入模式创建或覆盖文件
with open('output-DNN_features_spearman_all_videos.txt', 'w') as file:
    # 将列表1写入文件
    file.write("similarity_total_0:\n")
    for item in similarity_total_0:
        file.write(str(item) + "\n")

    # 写入一个空行
    file.write("\n")

    # 将列表2写入文件
    file.write("similarity_total_1:\n")
    for item in similarity_total_1:
        file.write(str(item) + "\n")


plt.hist(similarity_total_0, bins =30, alpha = 0.5, color='#1BA1E2', label='Real')
plt.hist(similarity_total_1, bins =30, alpha = 0.5, color='#FF3333', label='Fake')
plt.legend()
plt.legend(loc='upper left')
plt.xlabel("cosine similarity")
plt.ylabel("number")
plt.savefig("inter-frame_correlation.png")

cap.release()

