#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
import glob
from tqdm import tqdm
import json
from facenet_pytorch import MTCNN
import torch
from model import ECL
from scipy.stats import spearmanr
from data.transform import get_transforms
from lib.test_util import get_crop
from sklearn.cluster import KMeans
from collections import Counter


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, help='The path of test data.', default=None)
    parser.add_argument('--checkpoint_path', type=str, help='The path of trained encoder.',
                        default=None)
    parser.add_argument('--output_path', type=str, help='The path of output files.', default=None)
    args = parser.parse_args()
    return args


def get_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ECL()

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.cuda()

    return model


def get_features(cropped_face, encoder):
    test_transform = get_transforms(name="val")
    input = test_transform(cropped_face)

    input = np.reshape(input, (1, input.shape[0], input.shape[1], input.shape[2]))
    input = input.cuda(non_blocking=True)
    with torch.no_grad():
        output, _ = encoder(input)
    output = output.cpu()
    output = output.detach().numpy()

    return output


def main():
    args = args_func()
    test_data_path = args.test_data_path
    output_path = args.output_patha
    test_data_list = glob.glob(os.path.join(test_data_path, '*'))
    video_extensions = ['.mp4', '.mkv', '.avi']
    test_data_list = [file for file in test_data_list if
                      os.path.splitext(file.lower())[1] in video_extensions]
    # test_data_name_list = list(map(lambda x: os.path.basename(x), test_data_list))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mtcnn = MTCNN(device='cuda:0').eval()

    encoder = get_model(args)
    encoder.eval()
    features_total = []
    correlation_total = []
    total_files = len(test_data_list)
    with tqdm(total=total_files, desc='Processing files', unit='file') as pbar:
        for process_data in test_data_list:
            cap = cv2.VideoCapture(process_data)

            video_name = os.path.splitext(os.path.basename(process_data))[0]
            video_name = f"{video_name}"

            video_with_features = []

            prev_face_descriptor = None

            correlation_sum = []
            correlation = 0

            v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for j in range(v_len):
                ret, frame = cap.read()

                if ret:
                    cropped_face = get_crop(frame, mtcnn)
                    output = get_features(cropped_face, encoder)

                    video_with_features.append({"video_name": video_name, "features": output})

                    if prev_face_descriptor is not None:
                        correlation, p_value = spearmanr(output[0], prev_face_descriptor[0])

                    prev_face_descriptor = output
                    # print(similarity)
                    correlation_sum.append(correlation)

            filtered_data = [x for x in similarity_sum if x != 0]
            similarity_sum = filtered_data
            total = sum(correlation_sum)
            if len(correlation_sum) < 3:
                continue
            correlation_avg = total / len(correlation_sum)
            video_with_correlation = {"video_name": video_name, "correlation": correlation_avg}

            features_total.extend(video_with_features)
            correlation_total.append(video_with_correlation)
            pbar.update(1)

        cap.release()

    features = np.array([item['features'].flatten() for item in features_total])

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)

    image_labels = {}

    for i, item in enumerate(features_total):
        video_id = item['video_name']
        cluster_label = labels[i]

        if video_id not in image_labels:
            image_labels[video_id] = []

        image_labels[video_id].append(cluster_label)

    video_labels = {}

    for video_id, labels_list in image_labels.items():
        counter = Counter(labels_list)
        most_common_label = counter.most_common(1)[0][0]
        video_labels[video_id] = most_common_label

    video_correlation = {item["video_name"]: item["correlation"] for item in correlation_total}

    keys_for_0 = [key for key, value in video_labels.items() if value == 0]
    values_for_0 = [video_correlation[key] for key in keys_for_0]

    keys_for_1 = [key for key, value in video_labels.items() if value == 1]
    values_for_1 = [video_correlation[key] for key in keys_for_1]

    average_for_0 = sum(values_for_0) / len(values_for_0)
    average_for_1 = sum(values_for_1) / len(values_for_1)

    if average_for_0 > average_for_1:
        fake_keys = keys_for_0
        real_keys = keys_for_1
    else:
        fake_keys = keys_for_1
        real_keys = keys_for_0

    for key in fake_keys:
        video_labels[key] = "FAKE"

    for key in real_keys:
        video_labels[key] = "REAL"

    json_data = [{"video_name": key, "pred_label": value} for key, value in video_labels.items()]

    with open('test.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

