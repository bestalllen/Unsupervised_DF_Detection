# # # # # read csv文件

import csv
from sklearn import metrics
import os
import sys
import shutil
def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        f_folder = fpath.split('/')[-1]
        f_type = fpath.split('/')[-1]
        f_name = f_type + "_" + fname
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + f_name)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + f_name))

path= '/home/xsc/experiment/Exploiting-Visual-Artifacts-master/VA_UADFV.csv'
pesudo_fake = []
pesudo_real = []
with open(path) as f:
    reader = csv.reader(f)
    column=[row for row in  reader]
    for i in range(len(column)):
        if column[i][2] == '1':
            pesudo_fake.append(column[i][0])
        elif column[i][2] == '0':
            pesudo_real.append(column[i][0])
print(len(pesudo_fake))
print(len(pesudo_real))
# import glob
# import random
# train_real = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/Celeb-DF/val/0_real/*')
# train_fake = glob.glob('/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/Celeb-DF/val/1_fake/*')
# train_fake = random.sample(train_fake, len(train_real))
# dstpath_fake='/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/Celeb-DF/val/new_1_fake/'
# for i in train_fake:
#     mycopyfile(i, dstpath_fake)
#     print("copy success!")
# sys.exit()     
# # fake part
# pesudo_fake_video = []
# for i in pesudo_fake:
#     f_path, f_name = os.path.split(i)
#     video = f_name.split('_')[0] + '_' + f_name.split('_')[1]+'_'+f_name.split('_')[2] + '_' + f_name.split('_')[3]
#     pesudo_fake_video.append(video)

# pesudo_fake_video_set = set(pesudo_fake_video)
# print(len(pesudo_fake_video_set))
# confid_pesudo_fake_video = []
# num= 0
# for item in pesudo_fake_video_set:
#     if pesudo_fake_video.count(item) >= 16:
#         confid_pesudo_fake_video.append(item)

# fake = 0
# real = 0
# for i in confid_pesudo_fake_video:
#     if i.split('_')[0] == '1':
#         fake +=1
#     else:
#         real+=1 
# print(fake)
# print(real)
# print(len(confid_pesudo_fake_video))
# # sys.exit()
# # print(pesudo_real)
# import glob
# fake_root = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/DFDC/train/1_fake'
# real_root = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/DFDC/train/0_real'
# confid_pesudo_fake_images = []
# for i in confid_pesudo_fake_video:
#     if i.split('_')[0] == '1':
#         path = os.path.join(fake_root,'*'+ i + '*.png')
#     else:
#         path = os.path.join(real_root,'*'+ i + '*.png')
#     confid_pesudo_fake_images.append(path)
# confid_pesudo_fake_images_list = []
# for i in confid_pesudo_fake_images:
#     confid_pesudo_fake_images_list += glob.glob(i)


# print(len(confid_pesudo_fake_images_list))
# # sys.exit()


# # # # real part
# pesudo_real_video = []
# for i in pesudo_real:
#     f_path, f_name = os.path.split(i)
#     video = f_name.split('_')[0] + '_' + f_name.split('_')[1]+'_'+f_name.split('_')[2] + '_' + f_name.split('_')[3]
#     pesudo_real_video.append(video)

# pesudo_real_video_set = set(pesudo_real_video)
# confid_pesudo_real_video = []
# for item in pesudo_real_video_set:
#     if pesudo_real_video.count(item) >= 16:
#         confid_pesudo_real_video.append(item)

# # import random
# # confid_pesudo_real_video = random.sample(confid_pesudo_real_video, len(confid_pesudo_fake_video))
# confid_pesudo_real_images = []
# for i in confid_pesudo_real_video:
#     if i.split('_')[0] == '1':
#         path = os.path.join(fake_root, '*'+ i + '*.png')
#     else:
#         path = os.path.join(real_root, '*'+ i + '*.png')
#     confid_pesudo_real_images.append(path)

# confid_pesudo_real_images_list = []
# for i in confid_pesudo_real_images:
#     confid_pesudo_real_images_list += glob.glob(i)

# print(len(confid_pesudo_real_images_list))

# fake_ = 0
# real_ = 0
# for i in confid_pesudo_real_video:
#     if i.split('_')[0] == '1':
#         fake_ +=1
#     else:
#         real_+=1
# print(fake_)
# print(real_)
# print(len(confid_pesudo_real_video))
# print(len(confid_pesudo_real_images_list))

# import random
# confid_pesudo_fake_images_list = random.sample(confid_pesudo_fake_images_list, len(confid_pesudo_real_images_list)) # 为保持样本均衡
# # # # # print(len(pesudo_real_new))
dstpath_fake = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/UADFV/UADFV_acc663%/1_fake/'
for i in pesudo_fake:
    mycopyfile(i, dstpath_fake)
    print("copy success!")

dstpath_real = '/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/UADFV/UADFV_acc663%/0_real/'
for i in pesudo_real:
    mycopyfile(i, dstpath_real)
    print("copy success!")

from sklearn import metrics
import csv
path = '/home/xsc/experiment/Exploiting-Visual-Artifacts-master/VA_UADFV.csv'
with open(path) as f:
    reader = csv.reader(f)
    gt_column=[row[1] for row in  reader]
    # print(column)
    gt_delet_head_column = gt_column[1:]
    gt_column_new = [int(i) for i in gt_delet_head_column]
    # print(column_new)
with open(path) as f:
    reader = csv.reader(f)
    pred_column=[row[2] for row in  reader]
    # print(column)
    pred_delet_head_column = pred_column[1:]
    pred_column_new = [int(i) for i in pred_delet_head_column]
    # print(len(pred_column_new))


test_acc = metrics.accuracy_score(gt_column_new, pred_column_new)
print(f'test_acc:{test_acc}')