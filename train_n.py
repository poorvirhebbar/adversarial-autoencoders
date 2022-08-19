#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
#import seaborn as sns

#torch.manual_seed(43)
#np.random.seed(42)

from util import *
from data_n import *
from data_n import getBreakhisDataset
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
#from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torchvision.utils import make_grid





def train(enc, dec, disc, clsfr, dataset, device):

    """
    enc, dec, disc, and clsfr are networks
    dataset["train"] and dataset["eval"] can be used
    device is either CPU or GPU
    """
    least_loss = np.Inf
    historytrain = []
    historyeval = []
  # hyperparameters
    epochs = 1
    batch_size = 32
    lamda1 = 1.0 # classifier
    lamda2 = 1.0 # discriminator
    variance = 0.1

    weighted = True # weighted updates for clsfr
    binary = True # [0-1 weighting]
    thresh = 0.01 # threshold on loss value [for binary weighting]
    beta = 1e-5 # for multiplication in gaussian num
    w_n = True # weight normalization

  # Loss Functions
    DiscLoss_1 = nn.CrossEntropyLoss().to(device)
    DiscLoss_2 = None # placeholder
    ClsfrLoss = None # placeholder
    RecreationLoss = None # placeholder
    if weighted is True:
        DiscLoss_2 = nn.CrossEntropyLoss(reduction='mean').to(device)
        ClsfrLoss = nn.CrossEntropyLoss(reduction='none').to(device)
        RecreationLoss = nn.MSELoss(reduction='none').to(device)
    else:
        DiscLoss_2 = nn.CrossEntropyLoss(reduction='mean').to(device)
        ClsfrLoss = nn.CrossEntropyLoss(reduction='mean').to(device)
        RecreationLoss = nn.MSELoss(reduction='mean').to(device)

  # Optimizers
    disc_optim = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
    main_optim = optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(clsfr.parameters()), lr=0.0002, betas=(0.5, 0.999))
    
    weights = make_weights_for_balanced_classes(dataset["train"], 4)                                                              
    weights = torch.DoubleTensor(weights)
    #print("weights = ",weights,"size=",len(weights))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #print(dataset["train"])
    
    # get the data loader
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, sampler = sampler)

    # iterate for epochs
    
    for epoch in range(1, epochs+1):

    # set the flags to train
        enc.train()
        dec.train()
        disc.train()
        clsfr.train()

    # get the data loader
        #dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, sampler = sampler)

    # initialize loss values
        loss_disc_1 = 0.
        loss_disc_2 = 0.
        loss_clsfr = 0.
        loss_rec = 0.
        correct_1 = 0 # disc_1
        correct_2 = 0 # disc_2
        correct_3 = 0 # clsfr
        num_pts = 0

    # iterate over mini batches
        for data, target in dataloader:

      # put data and target to device
            data = data.to(device)
            #print("data form dataloader = ",data.shape)
            target = target.to(device)
            #print(target.dtype)
            #target[target == 0] = -3
      # data.shape = [n,3,l,b]
      # target.shape = [n]

      # TRAIN DISCRIMINATOR
      # 0 means it is from encoder
      # 1 means it is from prior

      # set gradients to zero
            disc_optim.zero_grad()
            enc.zero_grad()
            disc.zero_grad()

      # get hidden and reshape
            hidden = enc(data).view(data.shape[0], -1)
            #print("hidden shape = ",hidden.shape)
            #print("hidden = ",hidden,"hidden shape = ",hidden.shape)
            #print("hidden shape = ",hidden.shape)
            #print("hidden = ",hidden,"hidden shape = ",hidden.shape)
            h1 = np.array(hidden.detach().cpu())
            generated_vectors = gen_vectors(h1.shape)
            target_mapping = target_conversion(generated_vectors,target.cpu().tolist())
            #print("target mapped",target_mapping)
            #print("shape of target mapped",np.array(target_mapping).shape)
            target_mapping = torch.tensor(target_mapping).to(device)


      # sample prior according to target
            prior = (torch.randn(hidden.shape)*variance).to(device)
            #print("prior = ",prior,"prior shape = ",prior.shape)
            #print("target = ",target,"target shape = ",target.shape)
            prior = prior + target_mapping # just add the class index to mean
            #print("prior acc to classes = ",prior)
            
      # concatenate to get X and Y
            X = torch.cat([hidden, prior])
            Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

      # update X according to the target
      # append the one-hot vector of target

      # calculate the one-hot vector
            idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
            #print("idx = ",idx)
            OH = torch.zeros((len(target), dataset["train"].num_classes)).to(device)
            OH[idx[:,0], idx[:,1]] = 1
            #print("OH = ",OH)
            OH = torch.cat([OH,OH])

      # append to X
            X = torch.cat([X,OH], 1)
            #print("X = ",X)
            #print("Y = ",Y)
      # get output of discriminator
            out = disc(X)
            #print("out = ",out)

      # calculate loss and update params
            loss1 = DiscLoss_1(out, Y)
            loss1.backward()
            disc_optim.step()

      # get accuracy and loss_disc_1
            correct = torch.sum(Y == torch.argmax(out, 1))
            loss_disc_1 += len(X)*loss1
            correct_1 += correct

      # TRAIN ENCODER, DECODER, CLASSIFIER
      # 0 means it is from encoder
      # 1 means it is from prior

      # set gradients to zero
            main_optim.zero_grad()
            enc.zero_grad()
            dec.zero_grad()
            disc.zero_grad()

      # get hidden
            hidden = enc(data)
            #print(hidden.shape)

      # getting weights
            weights = None # placeholder
            if weighted is True:

                weights = hidden.view(hidden.shape[0], -1)
                weights = weights - (target_mapping)
                weights = beta*weights*weights
                weights = torch.sum(weights, 1)
                weights = torch.exp(-1*weights)

        # print(weights)

                if w_n is True:
                    weights = (weights / torch.sum(weights).float())*len(weights)

        # print(weights)

                if binary is True:
                    #print("binary used")
                    weights = (weights > thresh*len(weights)).float()
                    num_pts += torch.sum(weights)
                    weights = (weights / torch.sum(weights).float())*len(weights)
                    #print(weights)
          
        # print(weights)


      # add reconstruction error to loss
            data_ = dec(hidden)
            #print(data_.shape)
            loss2 = RecreationLoss(data, data_)
      # multiply by weights (if reqd)
            if weighted is True:
                loss2 = loss2.view(loss2.shape[0], -1)
                loss2 = torch.mean(loss2, 1)
                loss2 = torch.mean(loss2*weights)
            loss_rec += len(data)*loss2

      # reshape hidden
            hidden = hidden.view(data.shape[0], -1)
            #print(hidden.shape)
            #print(target.shape)

      # get output of classifier and calculate loss
            out1 = clsfr(hidden)
            loss3 = ClsfrLoss(out1, target)
      # multiply by weights (if reqd)
            if weighted is True:
                loss3 = torch.mean(loss3*weights)
            loss_clsfr += len(hidden)*loss3

      # get accuracy of classifier
            correct = torch.sum(target == torch.argmax(out1, 1))
            correct_3 += correct

      # sample prior according to target
            prior = (torch.randn(hidden.shape)*variance).to(device)
            prior = prior + target_mapping # just add the class index to mean

      # concatenate to get X and Y
            X = torch.cat([hidden, prior])
            Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

      # update Y according to the target
      # append the one-hot vector of target

      # # calculate the one-hot vector (NO NEED TO DO AGAIN)
      # idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
      # OH = torch.zeros((len(target), dataset["train"].num_classes))
      # OH[idx[:,0], idx[:,1]] = 1
      # OH = torch.cat([OH,OH])

      # append to X
            X = torch.cat([X,OH], 1)

      # get output of discriminator
            out2 = disc(X)

      # calculate disc loss
            loss4 = DiscLoss_2(out2, Y)
            loss_disc_2 += len(X)*loss4


            loss = loss2 + lamda1*loss3 - lamda2*loss4
            loss.backward()
            main_optim.step()

      # get accuracy and loss_disc_2
            correct = torch.sum(Y == torch.argmax(out2, 1))
            correct_2 += correct

            

        loss_disc_1 = loss_disc_1/(2*len(dataset["train"]))
        loss_disc_2 = loss_disc_2/(2*len(dataset["train"]))
        loss_clsfr = loss_clsfr/(len(dataset["train"]))
        loss_rec = loss_rec/len(dataset["train"])
        acc = correct_3*100.0/float(len(dataset["train"]))
        train_res = {'loss_epoch' : loss_clsfr, 'accuracy' : acc}
    
    # Pretty Printing
        if binary and weighted is True:
            print("Using %04d/%04d (%06f) points"%(num_pts, len(dataset["train"]), num_pts*100.0/float(len(dataset["train"]))))
        print("[Disc1] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%            (epoch, epochs, loss_disc_1, correct_1, 2*len(dataset["train"]), correct_1*50.0/float(len(dataset["train"]))))
        print("[Disc2] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%            (epoch, epochs, loss_disc_2, correct_2, 2*len(dataset["train"]), correct_2*50.0/float(len(dataset["train"]))))
        print("[Clsfr] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%            (epoch, epochs, loss_clsfr, correct_3, len(dataset["train"]), correct_3*100.0/float(len(dataset["train"]))))
        print("[Dec] Epoch %04d/%04d : Loss : %06f"%            (epoch, epochs, loss_rec))
        result = eval_model(enc, clsfr, dataset, device, "test")
        print()
        #print(result)
        eval_loss = result['loss_epoch']
        if eval_loss < least_loss :
            least_loss = eval_loss
            print('saving checkpoint for least loss = ',least_loss)
            #save_model(enc, "encleast_n1.pt")
            #save_model(dec, "decleast_n1.pt")
            #save_model(disc, "discleast_n1.pt")
            #save_model(clsfr, "clsfrleast_n1.pt")
        #train_res = eval_model(enc, clsfr, dataset, device, "train")
        historytrain.append(train_res) 
        historyeval.append(result)
    
    train_losses = [x.get('loss_epoch') for x in historytrain]
    val_losses = [x.get('loss_epoch') for x in historyeval]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    #plt.savefig("plots/loss_LL_n1.jpg")
    plt.show()

    train_losses = [x.get('accuracy') for x in historytrain]
    val_losses = [x.get('accuracy') for x in historyeval]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. No. of epochs')
    #plt.savefig("plots/accuracy_LL_n1.jpg")
    plt.show()
  # -------------------------------------------------- #

  # VISUALIZE
    visualize_embedding(enc, dataset, device, "test", "mdl_LL", weighted, binary, w_n, thresh, beta)

  # -------------------------------------------------- #

    return enc, dec, disc, clsfr




def plot_multiclass_roc(test_probs, y_test, n_classes, figsize=(17, 6)):
    y_score = np.array(test_probs)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    #sns.despine()
    #plt.savefig("ROC_n1")
    plt.show()






def main():
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = get_device(True)
    dataset = getBreakhisDataset()
    #print(dataset)

    conv = [3,4,8,16,32]
    fc = [32,16,4]
    shape = dataset["train"].shape

    enc = EncoderNetwork(conv, shape).to(device)
    #print(enc.out_p)
    dec = DecoderNetwork(conv[::-1], enc.out_p).to(device)
    #print(dec)
    disc = FullyConnectedNetwork(fc, enc.size+dataset["train"].num_classes).to(device) # to append classes
    #print("datset classes = ",dataset["train"].num_classes)
    clsfr = FullyConnectedNetwork(fc, enc.size).to(device)
    #print("enc size = ",enc.size)

    enc, dec, disc, clsfr = train(enc, dec, disc, clsfr, dataset, device)

    #save_model(enc, "enc100_LL_n1.pt")
    #save_model(dec, "dec100_LL_n1.pt")
    #save_model(disc, "disc100_LL_n1.pt")
    #save_model(clsfr, "clsfr100_LL_n1.pt")

    #enc.load_state_dict(torch.load("models/encleast_n1.pt"))
    #clsfr.load_state_dict(torch.load("models/clsfrleast_n1.pt"))

   # weights = make_weights_for_balanced_classes(dataset["eval"], 4)                                                              
   # weights = torch.DoubleTensor(weights)
   # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    test_dl = torch.utils.data.DataLoader(dataset["test"], batch_size=50) #sampler =sampler)
    
    result,test_probs, test_preds,testy_labels = test_model(enc, clsfr, dataset, device, "test")

    probs_for_4_classes = np.array(test_probs)

    #probs_save = {'Y' : testy_labels , 'class0' : probs_for_4_classes[:,0], 'class1' : probs_for_4_classes[:,1], 'class2' : probs_for_4_classes[:,2], 'class3' : probs_for_4_classes[:,3]}
    
    #dict_save = {'Expected' : testy_labels , 'predictions' : test_preds}# , 'dec_predictions' : dec_preds}
    
    #prediction = pd.DataFrame(dict_save, columns=['Expected','predictions']).to_csv('predictions_LL_n1.csv')

    #p = pd.DataFrame(probs_save, columns=['Y','class0','class1','class2','class3']).to_csv('p7n1.csv')
    
    plot_multiclass_roc(test_probs, testy_labels, n_classes=4, figsize=(16, 10))


main()






