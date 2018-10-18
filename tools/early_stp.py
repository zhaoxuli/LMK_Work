# -*- coding: utf-8 -*-
file_path = './gender_loss.txt'

ctx = open(file_path,'r').readlines()
[train_loss_lst,evalu_loss_lst] = ctx

def if_stopping(train_loss_lst,evalu_loss,task_weight,k=5,eata=0.001):
    k_tra_lst = train_loss_lst[-k:].sort()
    k_tra_lst.median



