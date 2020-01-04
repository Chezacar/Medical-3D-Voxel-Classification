import argparse
import json
import os
import time
import numpy as np
import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from model.DnCNN import DnCNN
from model import Resnet
from model import VoxNet
from model import baseline
import pandas as pd
if __name__ == "__main__":
    time_start = time.time()
    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    #parser.add_argument("--epoch", default=28, type=int, help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = dataloader.test_set()
    #test_set.sort()
    test_loader = DataLoader.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"]) 
    model = VoxNet.VoxNet(2).to(DEVICE)
    # Test the train_loader
    path = './999.pkl'
    model.load_state_dict(t.load(path))
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        
        for batch_idx, [data,name] in enumerate(test_loader):
            data= data.to(DEVICE)
            out = model(data)
            out = out.squeeze()
            Name.append(name[0])
            Score.append(out[1].item())

    test_dict = {'Id':Name, 'Predicted':Score}
    test_dict_df = pd.DataFrame(test_dict)
    print(test_dict_df)
    # path = './'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    test_dict_df.to_csv('./submisson.csv', index=False)