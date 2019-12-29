'''
used to read the data from the data folder
'''
import torch as t
import torchvision as tv
from torchvision import transforms
from torch import nn
import os
from PIL import Image
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class data_set(t.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx
        self.config = json.load(open('config1.json'))
        self.data_root = self.config["Taining_Dir"]
        self.names = np.array(os.listdir(self.data_root))
        self.sort()
        self.names = self.names[idx]
        self.label_path = self.config['Label_Path']
        self.init_transform()
        self.read_label()

    def sort(self):
        d = self.names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.names = np.array(sorted_key_list)
        # print(self.data_names)

    def read_label(self):
        dataframe = pd.read_csv(self.label_path)
        data = dataframe.values
        self.label = data[:,1][self.idx]

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5)
        ])

    def __getitem__(self, index):
        # print(self.names[index].split('.')[0])
        data = np.load(os.path.join(self.data_root, self.names[index]))
        voxel = np.array(data['voxel'].astype(np.float32)/255)
        seg =  np.array(data['seg'].astype(np.float32))
        label = self.label.astype(np.float32)[index]
        #label = self.label[index]
        # data = np.expand_dims(seg, axis=0)
        data = (voxel*seg)#[34:66,34:66,34:66]
        # [temp1,temp2,temp3] = np.random.rand(3)
        # if temp1 < 0.5: 
        #     data = np.flipud(data)
        # elif temp1 > 0.5:
        #     data = np.fliplr(data)
        # if temp2 < 0.25: 
        #     data = np.rot90(data,k=1,axes=(1,2))
        # elif temp2 < 0.5:
        #     data = np.rot90(data,k=2,axes=(1,2))
        # elif temp2 < 0.75:
        #     data = np.rot90(data,k=3,axes=(1,2))
        # elif temp2 < 1:
        #     data = np.rot90(data,k=4,axes=(1,2))
        # temp3_int = int(np.round(23*temp3))
        #print(temp3_int,temp3_int.dtype)
        # data[temp3_int:temp3_int+6,temp3_int:temp3_int+6,temp3_int:temp3_int+6] = 1
        # #data = (voxel*seg).unsqueeze(0)
        # #print(data.shape,data.dtype)
        data = data.copy()
        data = self.transform(data).unsqueeze(0)
        #data = nn.functional.interpolate(data, scale_factor=2, mode='bilinear', align_corners=False)
        return data, label

    def __len__(self):
        return len(self.names)


class MyDataSet():
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config1.json'))
        self.data_root = self.config["Taining_Dir"]
        self.data_names = np.array(os.listdir(self.data_root))
        self.DEVICE = t.device(self.config["DEVICE"])
        self.sort()

    def sort(self):
        d = self.data_names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
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
        self.train_idx = idx[:(int)(length*p)]
        self.test_idx = idx[(int)(length*p):]

        self.train_set = data_set(self.train_idx)
        self.test_set = data_set(self.test_idx)
        return self.train_set, self.test_set
        
    def __len__(self):
        return len(self.data_names)

class test_set(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.test_root = self.config["Test_Dir"]
        self.test_names = os.listdir(self.test_root)
        self.DEVICE = t.device(self.config["DEVICE"])
        self.init_transform()

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def sort(self):
        d = self.test_names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.test_names = sorted_key_list

        # sorted_dict = map(lambda x:{x:(int)(os.path.splitext(x)[0].strip('candidate'))}, d)
        # print(sorted_dict)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.test_root, self.test_names[index]))
        voxel = self.transform(data['voxel'].astype(np.float32))/255
        seg =  self.transform(data['seg'].astype(np.float32))
        #data = np.stack([voxel, seg], axis=0)
        #data = (voxel*seg).unsqueeze(0)
        data = (voxel*seg).unsqueeze(0)#[35:67,35:67,35:67].unsqueeze(0)
        #print('data',data.shape)
        #data = nn.functional.interpolate(data, scale_factor=2, mode='bilinear', align_corners=False)
        name = os.path.basename(self.test_names[index])
        name = os.path.splitext(name)[0]
        return data, name

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":

    # DataSet = MyDataSet()
    # train_set, test_set = DataSet.test_trian_split()
    # wild = In_the_wild_set()
    # print(len(train_set))
    # print(len(test_set))
    # print(train_set[0][0].shape)
    # print(test_set[0][0].shape)
    # print(wild[0].shape)
    # test_data = TestingData()
    # for i in range(len(train_data)):
    #     img, label = train_data[i]
    #     tv.transforms.ToPILImage()(img).save('result/input.jpg')
    #     tv.transforms.ToPILImage()(label).save('result/test.jpg')
    kf = KFold(n_splits=2)
    a = np.arange(100)
    print(kf.get_n_splits(a))
    for idx, [train_index, test_index] in enumerate(kf.split(a)):
        print(idx)
        print("TRAIN:", train_index, "TEST:", test_index)