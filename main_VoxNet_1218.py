import json
import time
import os
import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing
from model.dataloader import *
from model.DnCNN import DnCNN
from model.VoxNet1219 import VoxNet
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

if __name__ == "__main__":
    time_start = time.time()
    config = json.load(open("config.json"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']
    lam = config['Lam']
    date = '1224_mixup_rotation'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=0, type=int,
                        help="The epoch to be tested")
    parser.add_argument("--name", default='VoxNet(150epoch)', type=str,
                        help="Whether to test after training")
    args = parser.parse_args()

    DataSet = MyDataSet()

    # using K-fold
    kf = KFold(n_splits=5)
    idx = np.arange(len(DataSet))
    print(kf.get_n_splits(idx))
    for K_idx, [train_idx, test_idx] in enumerate(kf.split(idx)):
        writer = SummaryWriter('runs/{}_{}_{}_{}_Fold'.format(args.name,date, K_idx+1, LR))
        train_data, test_data = data_set(train_idx), data_set(test_idx)
        train_loader1 = DataLoader.DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        train_loader2 = DataLoader.DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        test_loader = DataLoader.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

        model = VoxNet(2).to(DEVICE)

        # if K_idx != 0:
        #     path = './saved_model/{}_{}_{}_{}_folds/{}.pkl'.format(args.name, K_idx,LR,date,EPOCH-1)
        #     print(path)
        #     model.load_state_dict(t.load(path))
        # Multi GPU setting
        # model = t.nn.DataParallel(model,device_ids=[0,1])

        optimizer = t.optim.SGD(model.parameters(), lr=LR,momentum=0.8)
        #optimizer = t.optim.Adam(model.parameters())
        criterian = t.nn.BCELoss().to(DEVICE)

        # Test the train_loader
        for epoch in range(args.epoch, EPOCH):
            #print(DEVICE)
            model = model.train()
            train_loss = 0
            correct = 0
            for [data1, label1],[data2,label2] in zip(train_loader1,train_loader2):
                data = lam * data1 + (1 - lam) * data2
                #print(data.size())
                label = lam * label1 + (1 - lam) * label2
                label_decision = label1.long()
                data, label, label_decision= data.to(DEVICE), label.to(DEVICE), label_decision.to(DEVICE)
                out = model(data)#.squeeze()[:,0]
                #print(out.squeeze()[:,0])
                loss = criterian(out.squeeze()[:,1], label)
                #print(out,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                #print(out)
                
                #print(label)
                #print(correct)
                pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(label_decision.view_as(pred)).sum().item()
                '''
                for i in range(out.size()[0]):
                    #print(out.size())
                    
                    if (out[i]-0.5) * (label[i]-0.5) > 0:
                        correct+=1
                '''
                #print(correct)
            train_loss /= len(train_loader1.dataset)
            train_acc = 100. * correct / len(train_loader1.dataset)
            writer.add_scalar('Training/Training_Loss', train_loss, epoch)
            writer.add_scalar('Training/Training_Acc', train_acc, epoch)
            print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, train_loss, correct, len(train_loader1.dataset), train_acc))

            model = model.eval()

            with t.no_grad():
                # Test the test_loader
                test_loss = 0
                correct = 0
                for batch_idx, [data, label] in enumerate(test_loader):
                    data, label = data.to(DEVICE), label.to(DEVICE)
                    out = model(data)
                    #print(out.size())
                    test_loss += criterian(out[0,1], label)
                    label = label.long()
                    pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
                    correct += pred.eq(label.view_as(pred)).sum().item()
                    '''
                    for i in range(out.size()[0]):
                        if (out[i]-0.5) * (label[i]-0.5) > 0:
                            correct+=1
                    '''
                # store params
                for name, param in model.named_parameters():
                    writer.add_histogram(
                        name, param.clone().cpu().data.numpy(), epoch)

                test_loss /= len(test_loader.dataset)
                test_acc = 100. * correct / len(test_loader.dataset)
                writer.add_scalar('Testing/Testing_Loss', test_loss, epoch)
                writer.add_scalar('Testing/Testing_Acc', test_acc, epoch)
                print('Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, test_loss, correct, len(test_loader.dataset), test_acc))
            save_model(model, epoch, '{}_{}_{}_{}_folds'.format(args.name, K_idx+1,LR,date))
            # eval_model_new_thread(epoch, 0)
            # LZX pls using the following code instead
            # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
            # multiprocess_idx += 1
        writer.close()

    time_end = time.time()
    print('time cost', time_end-time_start)
