# 自定义dataset
import os
from torch.utils import data
from torch.utils.data import dataset, dataloader
from PIL import Image
from torch.utils.data.dataset import Dataset
class FolderDataset(dataset.Dataset):
    def __init__(self, img_folder_root, transform=None):
        self.img_folder_root = img_folder_root
        self.transform = transform
        img_folder_list = os.listdir(self.img_folder_root)
        # self.length = 0
        self.img_path_list = []
        # 两级目录 eg:/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/FF++part/0_original/1.png
        for i in img_folder_list: 
            img_folder_path = os.path.join(self.img_folder_root, i)
            img_list = sorted(os.listdir(img_folder_path))
            for j in img_list:
                img_path = os.path.join(img_folder_path, j)
                self.img_path_list.append(img_path)
        # 三级目录 eg:/home/xsc/experiment/DFGC/train_FF++_crop/deepfake/000_003/1.png
        # for i in img_folder_list: 
        #     img_folder_path = os.path.join(self.img_folder_root, i)
        #     img_list_folder = sorted(os.listdir(img_folder_path))
        #     for j in img_list_folder:
        #         img_list_path = os.path.join(img_folder_path, j)
        #         img_folder = sorted(os.listdir(img_list_path))
        #         for img in img_folder:
        #             img_path = os.path.join(img_list_path, img)
        #             self.img_path_list.append(img_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        read_img_path = self.img_path_list[idx]
        img = Image.open(read_img_path)
        # print(read_img_path)
        # 二级目录 
        if read_img_path.split('/')[-2] == '0_real':
            label = 0
        # 三级目录
        # if read_img_path.split('/')[-3] == 'Celeb-real':
        #     label = 0
        else:
            label = 1
        img = self.transform(img)
        return read_img_path, img, label
        
class select_FolderDataset(dataset.Dataset):
    def __init__(self, select_img_path_list, new_assign_label, transform=None):
        self.img_path_list = select_img_path_list
        self.label_list = new_assign_label
        self.transform = transform
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        read_img_path = self.img_path_list[idx]
        # print(read_img_path)
        label = int(self.label_list[idx])
        # print(type(label))
        img = Image.open(read_img_path)
        # print(read_img_path)
        # 二级目录 
        # if read_img_path.split('/')[-2] == '0_real':
        #     label = 0
        # else:
        #     label = 1
        img = self.transform(img)

        return read_img_path, img, label

class video_dataset(Dataset):
    def __init__(self,video_root,transform = None):

        self.video_root = video_root
        video_folder_list = os.listdir(self.video_root)
        # self.labels = labels
        self.video_names = []
        for i in video_folder_list:
            video_path = os.path.join(self.video_root, i)
            self.video_names.append(video_path)
        self.transform = transform

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
    
        if video_path.split('/')[-2]=='1_fake':
            # print(video_path.split('/')[-2])
            label = 1
        else:
            label = 0
        frame_folder = os.listdir(video_path)
        frame_path_list = []
        for i in frame_folder:
            frame_path = os.path.join(video_path, i)
            frame_path_list.append(frame_path)
        for i,frame in enumerate(frame_path_list):
            frame = Image.open(frame)
            frames.append(self.transform(frame))
        frames = torch.stack(frames)
        #print("length:" , len(frames), "label",label)
        return frames, label
    
