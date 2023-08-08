# -*- coding: UTF-8 -*-
import os
import sys
import argparse
from collections import defaultdict
import dlib
import cv2
# from sklearn.externals import joblib
import numpy as np
import pandas as pd
import glob
from pipeline import face_utils
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import scipy.spatial
import pywt
from PIL import Image
def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input',
                        help='Path to input image or folder containting multiple images.', default='/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_deepfake_raw_all/train/1_fake/*.png')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',
                        default='./output')
    parser.add_argument('-p', '--pipeline', choices=['gan', 'deepfake', 'face2face'], dest='pipeline', default='deepfake')
    parser.add_argument('-f', '--features', dest='save_features', action='store_true',
                        help='Set flag to save features, e.g. to fit classifier.',
                        default=True)
    args = parser.parse_args()
    return args


# def load_classifiers(pipeline):
#     """Loads classifiers for specified pipeline."""
#     classifiers = None
#     if pipeline == 'gan':
#         classifiers = joblib.load('models/gan/bagging_knn.pkl')
#     elif pipeline == 'deepfake':
#         classifier_mlp = joblib.load('models/deepfake/mlp_df.pkl')
#         classifier_logreg = joblib.load('models/deepfake/logreg_df.pkl')
#         classifiers = [classifier_mlp, classifier_logreg]
#     elif pipeline == 'face2face':
#         classifier_mlp = joblib.load('models/face2face/mlp_f2f.pkl')
#         classifier_logreg = joblib.load('models/face2face/logreg_f2f.pkl')
#         classifiers = [classifier_mlp, classifier_logreg]
#     # else:
#     #     print 'Unknown pipeline argument.'
#     #     exit(-1)

#     return classifiers


def load_facedetector():
    """Loads dlib face and landmark detector."""
    # download if missing http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if not os.path.isfile('/home/xsc/experiment/LRNet-main/shape_predictor_68_face_landmarks.dat'):
        # print 'Could not find shape_predictor_68_face_landmarks.dat.'
        exit(-1) 
    face_detector = dlib.get_frontal_face_detector()
    sp68 = dlib.shape_predictor('/home/xsc/experiment/LRNet-main/shape_predictor_68_face_landmarks.dat')

    return face_detector, sp68

def distance(x, y, p=2):
    '''
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离,等于2时为欧氏距离
    output:distance(float):x到y的距离      
    '''   
    dis2 = np.sum(np.abs(x-y)**p) # 计算
    dis = np.power(dis2,1/p)
    return dis
import shutil
def mymovefile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        f_folder = fpath.split('/')[-1]
        f_name = f_folder + '_' + fname
        # f_name = "fromfake" + '_'+ fname
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, os.path.join(dstpath, fname))  # 复制文件
        print("move %s -> %s" % (srcfile, dstpath + f_name))
def main(input_path, output_path, pipeline, save_features):
    """Main function to process input files with selected pipeline.

    Given a path to single image or folder and a output path,
    images are processed with selected pipeline.
    Outputs are saved as .csv file. If selected, the computed feature vectors
    for classification are saved as .npy.
    The scores.csv file contains the output score of the different classifiers. The
    'Valid' value indicates if the face detection and segmentation was successful.

    Args:
        input_path: Path to image, or folder containing multiple images.
        output_path: Path to save outputs.
        pipeline: Selected pipeline for processing. Options: 'gan', 'deepfake', 'face2face'
        save_features: Boolean flag. If set true, feature vectors will be saved as single .npy file.
    """
    # transfer image names to list
    if os.path.isdir(input_path):
        file_list = [name for name in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, name))]
    else:
        file_list = [os.path.basename(input_path)]
        input_path = os.path.dirname(input_path)

    if len(file_list) == 0:
        # print 'No files at given input path.'
        exit(-1)

    # load classifiers, sanity check
    # classifiers = load_classifiers(pipeline)

    # setup face detector and landmark predictor
    face_detector, sp68 = load_facedetector()

    # create save folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result_dict = defaultdict(list)
    feature_vec_list = []
    label_list = []
    files_processed = 0
    # Celeb-DF
    # img_folder_1 = glob.glob('/home/xsc/experiment/DFGC/Celeb-DF-v2-crop/Celeb-real/*/*.png')
    # img_folder_1 =  random.sample(img_folder_1, 20)
    # img_folder_0 = glob.glob('/home/xsc/experiment/DFGC/Celeb-DF-v2-crop/Celeb-synthesis/*/*.png')
    # img_folder_0 =  random.sample(img_folder_0, 20)
    # DFD
    # img_folder_1 = glob.glob('/public_database/DFD/face/*/*/*.png')
    # print(len(img_folder_1))
    
    # img_folder_1 =  random.sample(img_folder_1, 10)
    img_folder_0 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/DFD/train/0_real/*.png')
    img_folder_0 =  random.sample(img_folder_0, 2000)

    # img_folder_1 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_FS_raw_all/train/1_fake/*.png')
    # img_folder_1 =  random.sample(img_folder_1, 10000)
    img_folder_2 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/DFD/train/1_fake/*.png')
    img_folder_2 =  random.sample(img_folder_2, 2000)
    # img_folder_3 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_F2F_raw_all/train/1_fake/*.png')
    # img_folder_3 =  random.sample(img_folder_3, 10000)
    # img_folder_4 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_NT_raw_all/train/1_fake/*.jpg')
    # img_folder_4 =  random.sample(img_folder_3, 10000)
    # img_folder_0 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++_deepfake_raw_all/train/0_real/*.png')
    # img_folder_0 =  random.sample(img_folder_0, 2000)
    # img_folder_0 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/train_360/1_fake/*')
    # img_folder_1 = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/train_360/0_real/*')
    # img_folder_0 =  random.sample(img_folder_0, 200)
    # img_folder_1 =  random.sample(img_folder_1, 200)
    img_folder = img_folder_0 + img_folder_2 
    img_path_list = []
    bad_img_count = 0
    bad_img_folder = "/home/xsc/experiment/PCL-I2G-main/F2G_data/F2G_clean/bad_img_folder/"
    for img_path in img_folder:
        # load image
        
        # if img_path.split('/')[-4]=='FF++_FS_raw_all':
        #     label = 0
        # elif img_path.split('/')[-4]=='FF++_deepfake_raw_all':
        #     label = 1
        # elif img_path.split('/')[-4]=='FF++_F2F_raw_all':
        #     label = 2    
        # elif img_path.split('/')[-4]=='FF++_NT_raw_all':
        #     label = 3
        if img_path.split('/')[-2]=='0_real':
            label = 0
        elif img_path.split('/')[-2]=='1_fake':
            label = 1
        img = cv2.imread(img_path)
        if img is None or img is False:
            # print "Could not open image file: %s" % os.path.join(input_path, input_file)
            # shutil.move(img_path, bad_img_folder)
            # print(img_path)
            # bad_img_count += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # from skimage import feature as ft
        # img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # feature_vectors = ft.hog(img_gray)
        # feature_vectors = feature_vectors.tolist()
        # print(len(feature_vectors))
        # print(type(feature_vectors))
        # feature_vectors = ft.hog(img_gray,orientations=6,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=False)
        # sys.exit()
        # wave = pywt.dwt2(img, 'haar', mode='periodization')
        # cA, (cH, cV, cD) = wave
        # waveplus = np.empty([cA.shape[0], cA.shape[1], 3])
        # waveplus[:, :, 0] = cH[:, :, 0]
        # waveplus[:, :, 1] = cV[:, :, 0]
        # waveplus[:, :, 2] = cD[:, :, 0]
        # waveplus = Image.fromarray(np.uint8(waveplus)).convert('RGB')
        # img = waveplus

        # if pipeline == 'deepfake':
            # if pipeline == 'face2face':
            # extend_roi = 0.1
            # else:
            #     extend_roi = 0.0
            # detect and crop faces
        extend_roi = 0.0
        face_crops, final_landmarks= face_utils.get_crops_landmarks(face_detector, sp68, img,
                                                                        roi_delta=extend_roi)
        # print(final_landmarks)
        if len(final_landmarks)==0:
            # shutil.move(img_path, bad_img_folder)
            bad_img_count += 1
            continue

        # # from pipeline import eyecolor
        # # # feature_vectors, valid_seg = eyecolor.process_faces(face_crops,final_landmarks, scale=768)
        # # # call feature extraction, classifier pipeline
        from pipeline import texture_v1
        feature_vectors, valid_seg, flag = texture_v1.process_faces(face_crops,final_landmarks,scale=256)
        # if flag==False:
        #     continue
    
        # print("feature vectors:")
        # print(feature_vectors)
        # print("valid_seg:")
        # print(valid_seg)
        # print(type(feature_vectors))
        # sys.exit()
        # feature_vectors = feature_vectors.tolist()
        # 加载视频
        # _,image_name = os.path.split(img_path)
        # if label == 0:
        #     video_name = image_name.split("_")[2]+".mp4"
        #     video_path = os.path.join("/public_database/FaceForensics++/original_sequences/youtube/raw/videos",video_name)
        # else:
        #     video_name = image_name.split("_")[2]+"_"+image_name.split("_")[3]+".mp4"
        #     video_path = os.path.join("/public_database/FaceForensics++/manipulated_sequences/Deepfakes/raw/videos",video_name)
        # cap = cv2.VideoCapture(video_path)
        # print(video_path)
        # # 初始化嘴唇特征向量
        # lip_features = []

        # # 读取第一帧
        # ret, prev_frame = cap.read()
        # prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # while True:
        #     # 读取当前帧
        #     ret, curr_frame = cap.read()
        #     if not ret:
        #         break
        #     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        #     # 检测人脸
        #     faces = face_detector(curr_gray)

        #     for face in faces:
        #         # 检测嘴唇特征点
        #         landmarks = sp68(curr_gray, face)
        #         lip_points = []
        #         for n in range(48, 68):
        #             x = landmarks.part(n).x
        #             y = landmarks.part(n).y
        #             lip_points.append((x, y))

        #         # 计算嘴唇运动特征
        #         lip_features.append(np.array(lip_points).flatten())

        #         # 绘制嘴唇区域
        #         # cv2.polylines(curr_frame, [np.array(lip_points)], True, (0, 255, 0), 2)

        #     # 计算光流
        #     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        #     # 提取嘴唇区域的光流特征
        #     lip_flow = flow[lip_points[0][1]:lip_points[6][1], lip_points[0][0]:lip_points[12][0], :]

        #     # 将嘴唇区域的光流特征转换为特征向量
        #     lip_features.append(lip_flow.flatten())

        #     # 更新前一帧
        #     prev_gray = curr_gray

        # # 将嘴唇特征向量级联成一个特征向量
        # feature_vectors = np.concatenate((feature_vectors, lip_features))

        feature_vec_list.append(feature_vectors)
        label_list.append(label)
        img_path_list.append(img_path)
        # print(feature_vec_list)
        # sys.exit()

        files_processed += 1
        print("Files processed: ", files_processed, " of ", len(img_folder))
    print("bad image num:")
    print(bad_img_count)
    # sys.exit()
    # print(feature_vec_list)
    print(len(feature_vec_list))
    # for i in feature_vec_list:
    #     print(len(i))
    # print(feature_vec_list)
    print(len(label_list))
    print(len(img_path_list))
    # sys.exit()
    tsne = TSNE(n_components=2,perplexity=1, init='pca')
    x_tsne = tsne.fit_transform(feature_vec_list)
    # tsne 归一化， 这一步可做可不做
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    tsne_norm = (x_tsne - x_min) / (x_max - x_min)

    # 根据label，把Fake和Real分开
    label_array = np.array(label_list, dtype='uint8')
    real_idxs = (label_array == 0)
    fake_idxs = (label_array == 1)
    tsne_real = tsne_norm[real_idxs]
    tsne_fake = tsne_norm[fake_idxs]

    plt.figure(figsize=(3, 3))
    plt.scatter(tsne_real[:, 0], tsne_real[:, 1], s=6, color='#FF3333', marker='^', label='Fake')
    # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
    plt.scatter(tsne_fake[:, 0], tsne_fake[:, 1], s=6, color='#1BA1E2', marker='o', label='Real')
    plt.rcParams.update({'font.size':5})
    plt.legend(loc='upper right', labelspacing = 1)
    plt.show()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,hspace=0.1,wspace=0.1)
    plt.xticks([]),plt.yticks([])  #去除坐标轴
    plt.savefig("T-SNE_for_VAF.pdf")
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x_tsne)

    # 预测数据的标签
    y_pred = kmeans.labels_

    def calculate_purity(labels_true, labels_pred):
        # 计算混淆矩阵
        cm = confusion_matrix(labels_true, labels_pred)
        # 计算每个聚类簇的最大类别数量之和
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        return purity

    # 计算纯度
    purity = calculate_purity(label_list, y_pred)
    print("Purity:", purity)

    sys.exit()
    # 使用余弦距离进行聚类
    # from Bio.Cluster import kcluster
    # from sklearn import metrics
    # clusterid, error, nfound = kcluster(feature_vec_list, nclusters = 2, dist='s')
    # ACC = metrics.accuracy_score(label_list, clusterid)
    # print("ACC")
    # print(ACC)
    # ARI = metrics.adjusted_rand_score(label_list, clusterid)
    # print("ARI(调整兰德指数)")
    # print(ARI)

    # NMI = metrics.normalized_mutual_info_score(label_list, clusterid)
    # print("NMI(标准化互信息)")
    # print(NMI)
    # sys.exit()
     
    # from sklearn.cluster import KMeans,SpectralClustering, DBSCAN
    # from sklearn import metrics
    # clf = KMeans(n_clusters=2)
    # fit_clf=clf.fit(feature_vec_list)
    # clf.predict(feature_vec_list)
    # # print(clf.cluster_centers_)
    # # print(clf.cluster_centers_[0])
    # center_0 = clf.cluster_centers_[0]
    # # center_0 = np.array(center_0)
    # # center_0 = center_0[None] # 扩充维度
    # # print("----------")
    # # print(clf.cluster_centers_[1])
    # center_1 = clf.cluster_centers_[1]
    # # center_1 = np.array(center_1)
    # # center_1 = center_1[None] # 扩充维度
    # dis_list_0 = []
    # dis_list_1 = []
    # img_path_list_0 = []
    # img_path_list_1 = []
    # y_pred = clf.fit_predict(feature_vec_list)
    # for i, pred in enumerate(y_pred):
    #     # result_ = np.array(feature_vec_list[i][:])
    #     # result_ = result_[None]
    #     if pred == 0:
    #         dis = distance(center_0, feature_vec_list[i][:])
    #         # print(result)
    #         # dis = scipy.spatial.distance.cdist(center_0, result_,'cosine')    
    #         dis_list_0.append(dis)
    #         img_path_list_0.append(img_path_list[i])
    #     else:
    #         dis = distance(center_1, feature_vec_list[i][:])
    #         # dis = scipy.spatial.distance.cdist(center_1, result_,'cosine')
    #         dis_list_1.append(dis)
    #         img_path_list_1.append(img_path_list[i])

    # sorted_id_0 = sorted(range(len(dis_list_0)), key=lambda k: dis_list_0[k])
    # sorted_id_1 = sorted(range(len(dis_list_1)), key=lambda k: dis_list_1[k])

    # nn_id_0 = sorted_id_0[:500]
    # nn_id_1 = sorted_id_1[:500]

    # confid_img = []
    # for i in nn_id_0:
    #     confid_img.append(img_path_list_0[i])
    # for j in nn_id_1:
    #     confid_img.append(img_path_list_1[j])
    # # print(confid_img)
    # y_trues = []
    # y_preds = len(nn_id_0) * [0] + len(nn_id_1) * [1]
    # for i in confid_img:
    #     # if i.split('/')[-1].split('_')[0] == '1':
    #     #     y_trues.append(1)
    #     # else:
    #     #     y_trues.append(0)
    #     if i.split('/')[-2].split('_')[0] == '1':
    #         y_trues.append(1)
    #     else:
    #         y_trues.append(0)
    # test_acc = metrics.accuracy_score(y_trues, y_preds)
    # print(f'test_acc:{test_acc}')
    # ACC = metrics.accuracy_score(label_list, y_pred)
    # print("ACC")
    # print(ACC)
    # sys.exit()
    print("欧氏距离聚类")
    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
    from sklearn import metrics
    kmeans = KMeans(n_clusters=2, init='k-means++')
    # print(kmeans)
    y_pred = kmeans.fit_predict(feature_vec_list)
    print(len(y_pred))
    AUC = metrics.roc_auc_score(label_list, y_pred)
    print("AUC")
    print(AUC)
    ACC = metrics.accuracy_score(label_list, y_pred)
    print("ACC")
    print(ACC)
    ARI = metrics.adjusted_rand_score(label_list, y_pred)
    print("ARI(调整兰德指数)")
    print(ARI)

    NMI = metrics.normalized_mutual_info_score(label_list, y_pred)
    print("NMI(标准化互信息)")
    print(NMI)
    sys.exit()
    # print("-----------------")
    # clustering = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(feature_vec_list)
    # y_pred = clustering.labels_
    # print("层次聚类：")
    # ACC = metrics.accuracy_score(label_list, y_pred)
    # print("ACC")
    # print(ACC)
    # ARI = metrics.adjusted_rand_score(label_list, y_pred)
    # print("ARI(调整兰德指数)")
    # print(ARI)
    # NMI = metrics.normalized_mutual_info_score(label_list, y_pred)
    # print("NMI(标准化互信息)")
    # print(NMI)
    # sys.exit()
    # 写入csv
    def pd_tocsv(file_path, y_trues, y_preds, sava_path=None):  # pandas库储存数据到excel
        dfData = {  # 用字典设置DataFrame所需数据
            '文件路径': file_path,
            '真实标签': y_trues,
            '分类结果': y_preds
        }
        df = pd.DataFrame(dfData)  # 创建DataFrame
        csv_file_Name = os.path.join(os.getcwd(), sava_path)
        df.to_csv(csv_file_Name, index=False)  # 存表，去除原始索引列（0,1,2...）

    sava_path = 'VA_less_sample.csv'
    pd_tocsv(file_path=img_path_list, y_trues=label_list, y_preds=y_pred, sava_path=sava_path)
    
if __name__ == '__main__':
    args_in = parse_args()
    main(args_in.input, args_in.output, args_in.pipeline, args_in.save_features)

