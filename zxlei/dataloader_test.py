import torch as t
import torchvision as tv
from torchvision import transforms
import os
from PIL import Image
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class data_set(t.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Test_Dir"]
        self.names = np.array(os.listdir(self.data_root))
        self.sort()
        self.names = self.names[idx]
        #self.label_path = self.config['Label_Path']
        self.init_transform()
        self.read_label()

    def sort(self):
        d = self.names
        sorted_key_list = sorted(d, key=lambda x: (int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.names = np.array(sorted_key_list)
        # print(self.data_names)

    '''def read_label(self):
        dataframe = pd.read_csv(self.label_path)
        data = dataframe.values
        self.label = data[:, 1][self.idx]'''

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # print(self.names[index].split('.')[0])
        data = np.load(os.path.join(self.data_root, self.names[index]))
        voxel = self.transform(data['voxel'].astype(np.float32)) / 255
        seg = self.transform(data['seg'].astype(np.float32))
        # label = self.label.astype(np.float32)[index]
        label = self.label[index]
        # data = np.expand_dims(seg, axis=0)
        data = (voxel * seg).unsqueeze(0)
        
        return data, label

    def __len__(self):
        return len(self.names)


class MyDataSet():
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.data_names = np.array(os.listdir(self.data_root))
        self.DEVICE = t.device(self.config["DEVICE"])
        self.gray = self.config["gray"]
        self.sort()

    def sort(self):
        d = self.data_names
        sorted_key_list = sorted(d, key=lambda x: (int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.data_names = np.array(sorted_key_list)
        # print(self.data_names)

    def test_trian_split(self, p=0.8):
        '''
        p is the portation of the training set
        '''
        length = len(self.data_names)

        # create a random array idx
        idx = np.array(range(length))
        np.random.shuffle(idx)
        self.train_idx = idx[:(int)(length * p)]
        self.test_idx = idx[(int)(length * p):]

        self.train_set = data_set(self.train_idx)
        self.test_set = data_set(self.test_idx)
        return self.train_set, self.test_set

    def __len__(self):
        return len(self.data_names)