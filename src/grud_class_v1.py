import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
import pandas as pd
from model import GRUD
from random import SystemRandom
import utils
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# from torch.utils.tensorboard import SummaryWriter
# experiment_id = 'classification'
# writer = SummaryWriter('runs/experiment_' + experiment_id)

import vessl
vessl.init()

inputpath = "./data/physionet/PhysioNet/raw/set-a/"
inputdict = {
    "ALP" : 0,             # o
    "ALT" : 1,             # o
    "AST" : 2,             # o
    "Albumin" : 3,         # o
    "BUN" : 4,             # o
    "Bilirubin" : 5,       # o
    "Cholesterol" : 6,     # o
    "Creatinine" : 7,      # o
    "DiasABP" : 8,         # o
    "FiO2" : 9,            # o
    "GCS" : 10,            # o
    "Glucose" : 11,        # o
    "HCO3" : 12,           # o
    "HCT" : 13,            # o
    "HR" : 14,             # o
    "K" : 15,              # o
    "Lactate" : 16,        # o
    "MAP" : 17,            # o
    "Mg" : 18,             # o
    "Na" : 19,             # o
    "PaCO2" : 20,          # o
    "PaO2" : 21,           # o
    "Platelets" : 22,      # o
    "RespRate" : 23,       # o
    "SaO2" : 24,           # o
    "SysABP" : 25,         # o
    "Temp" : 26,           # o
    "Tropl" : 27,          # o
    "TroponinI" : 27,      # temp: regarded same as Tropl
    "TropT" : 28,          # o
    "TroponinT" : 28,      # temp: regarded same as TropT
    "Urine" : 29,          # o
    "WBC" : 30,            # o
    "Weight" : 31,         # o
    "pH" : 32,             # o
    "NIDiasABP" : 33,      # unused variable
    "NIMAP" : 34,          # unused variable
    "NISysABP" : 35,       # unused variable
    "MechVent" : 36,       # unused variable
    "RecordID" : 37,       # unused variable
    "Age" : 38,            # unused variable
    "Gender" :39,          # unused variable
    "ICUType" : 40,        # unused variable
    "Height": 41           # unused variable
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(learning_rate, learning_rate_decay, n_epochs):
    # def df_to_x_m_d(df, inputdict, mean, std, size, id_posistion, split):
    size = 49 # steps ~ from the paper
    id_posistion = 37
    input_length = 33 # input variables ~ from the paper
    
    if os.path.exists("./input/x_mean_aft_nor.npy") and os.path.exists("./input/dataset.npy"):
        pass
    else:
        dataset = np.zeros((1,3, input_length, size))

        all_x_add = np.zeros((input_length,1))

        filenames = os.listdir(inputpath)
        filenames.sort(key=lambda x: int(x[:-4]))

        for filename in filenames:
            df = pd.read_csv(inputpath + filename,\
                            header=0,\
                            parse_dates=['Time'],\
                            date_parser=utils.timeparser)
            s_dataset, all_x, id = utils.df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion, split=input_length)
            
            dataset = np.concatenate((dataset, s_dataset[np.newaxis, :,:,:]))
            all_x_add = np.concatenate((all_x_add, all_x), axis=1)
            

        dataset = dataset[1:, :,:,:]    
        # (total datasets, kind of data(x, masking, and delta), input length, num of varience)
        # (4000, 3, 33, 49)
        all_x_add = all_x_add[:, 1:]

        train_proportion = 0.8
        train_index = int(all_x_add.shape[1] * train_proportion)
        train_x = all_x_add[:, :train_index]

        x_mean = utils.get_mean(train_x)
        x_std = utils.get_std(train_x)

        x_mean = np.asarray(x_mean)
        x_std = np.asarray(x_std)

        dataset = utils.dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)

        nor_mean, nor_median, nor_std, nor_var = utils.normalize_chk(dataset)
        x_mean = nor_mean
        np.save('./input/x_mean_aft_nor', nor_mean)
        # np.save('./input/x_median_aft_nor', nor_median)
        np.save('./input/dataset', dataset)


    A_outcomes = pd.read_csv('./data/physionet/PhysioNet/raw/Outcomes-a.txt')
    # y1_outcomes = utils.df_to_y1(A_outcomes)
    # np.save('./input/y1_out', y1_outcomes)

#    t_dataset = dataset
    t_dataset = np.load('./input/dataset.npy')
    x_mean = torch.Tensor(np.load('./input/x_mean_aft_nor.npy'))

    # t_out = np.load('./input/y1_out.npy')
    t_out = utils.df_to_y1(A_outcomes)

    train_dataloader, dev_dataloader, test_dataloader = utils.data_dataloader(t_dataset, t_out, train_proportion=0.8, dev_proportion=0.2)

    input_size = 33 # num of variables base on the paper
    hidden_size = 33 # same as inputsize
    output_size = 1
    num_layers = 49 # num of step or layers base on the paper
    print("checkpoint")
    #dropout_type : Moon, Gal, mloss
    model = GRUD(input_size = input_size, hidden_size= hidden_size, output_size=output_size, dropout=0, dropout_type='mloss', x_mean=x_mean, num_layers=num_layers, device = device)
    # load the parameters
    # model.load_state_dict(torch.load('./save/grud_para.pt'))
    # model.eval()
    count = utils.count_parameters(model)
    model = model.to(device)


    criterion = torch.nn.BCELoss()

    epoch_losses = []
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    
    for epoch in range(n_epochs):
        
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        # losses, acc = losses.to(device), acc.to(device)
        label, pred = [], []
        # label, pred = label.to(device), pred.to(device)
        y_pred_col= []
        # y_pred_col = y_pred_col.to(device)
        model.train()
        for train_data, train_label in train_dataloader:

            train_data = train_data.to(device)
            train_label = train_label.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            train_data = torch.squeeze(train_data)
            train_label = train_label.squeeze(-1)
            train_data = train_data.to(torch.float32)
            # Forward pass : Compute predicted y by passing train data to the model            
            y_pred = model(train_data)

            # y_pred = y_pred[:, None]
            # train_label = train_label[:, None]
            
            #print(y_pred.shape)
            #print(train_label.shape)
            
            # Save predict and label
            # for pred_i in y_pred:
            #     # 배치 내 각 예측값을 Python 스칼라 값으로 변환하여 리스트에 추가
            #     y_pred_col.append(pred_i.item())
            #     pred.append(pred_i.item() > 0.5)
            # # y_pred_col.append(y_pred.item())
            # pred.append(y_pred.item() > 0.5)
            # for label_i in train_label:
            #     label.append(label_i.item())
            # label.append(train_label.item())
            # y_pred_col += y_pred.tolist()
            pred += (y_pred > 0.5).tolist()
            label += train_label.tolist()
            
            #print('y_pred: {}\t label: {}'.format(y_pred, train_label))
            # print("pred_y", y_pred.shape, "train", train_label.shape)


            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        # train_pred_out = pred
        # train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        
        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            dev_data = dev_data.to(device)
            dev_label = dev_label.to(device)
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            dev_data = torch.squeeze(dev_data)
            dev_label = dev_label.squeeze(-1)
            dev_data = dev_data.to(torch.float32)

            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)
            
            # # Save predict and label
            # pred.append(y_pred.item())
            # label.append(dev_label.item())
            
            pred += y_pred.tolist()
            label += dev_label.tolist()

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.item())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label
        dev_auc = roc_auc_score(label, pred)

        vessl.log(step = epoch, payload ={'Loss/Val': dev_loss,
                                        'Accuracy/Val': dev_acc,
                                        'AUC/Val': dev_auc})
        
        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            test_data = test_data.to(device)
            test_label = test_label.to(device)

            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            test_label = test_label.squeeze(-1)
            test_data = test_data.to(torch.float32)

            
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)
            
            # # Save predict and label
            # pred.append(y_pred.item())
            # label.append(test_label.item())
            pred += y_pred.tolist()
            label += test_label.tolist()

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())
            
        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)
        
        test_pred_out = pred
        test_label_out = label
                
        # epoch_losses.append([
        #      train_loss, dev_loss, test_loss,
        #      train_acc, dev_acc, test_acc,
        #      train_pred_out, dev_pred_out, test_pred_out,
        #      train_label_out, dev_label_out, test_label_out,
        #  ])
        
        pred = np.asarray(pred)
        label = np.asarray(label)
        
        auc_score = roc_auc_score(label, pred)
        
        vessl.log(step = epoch, payload ={'AUC/Test': auc_score})
        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, auc_score))
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        # torch.save(model.state_dict(), './save/grud_mean_grud_para.pt')
        
        # print(train_log)
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 명령행 인수로 learning_rate, learning_rate_decay, n_epochs를 받습니다.
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lrd", type=int, default=10, help="Learning rate decay")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of epochs")

    args = parser.parse_args()

    # main 함수에 인수 전달
    main(args.lr, args.lrd, args.n_epochs)