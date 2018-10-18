# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import random

draw_flag = True 
get_loss_flag = False
log_file = '../log_files/shuffle_sgd_equal_train.log'
#log_file = '../log_files/New_points_train.log'
#log_file = '../log_files/cg_lr_points_train.log'
#[Train Epoch 0] All_loss: 5.510 gender_loss: 0.629 smile_loss: 0.649 glass_loss: 0.460 head_loss: 1.340 points_loss: 0.229283
#[Evalu Epoch 0] All_loss: 3.855 gender_loss: 0.667 smile_loss: 0.688 glass_loss: 0.334 head_loss: 1.595 points_loss: 0.0417
out_dir = 'Curve'+os.sep+log_file.split('/')[-1].split('.')[0]
start_loc =1

if os.path.exists(out_dir) ==False:
    os.makedirs(out_dir)

ctx = open(log_file,'r').readlines()

kinds_lst = ["All_loss","gender_loss",'smile_loss',"glass_loss","head_loss","points_loss"]
Train_loss = {
        "All_loss":[] ,
        "gender_loss":[] ,
        "smile_loss":[] ,
        "glass_loss":[] ,
        "head_loss":[] ,
        "points_loss":[] ,
        }
Evalu_loss = {
        "All_loss":[] ,
        "gender_loss":[] ,
        "smile_loss":[] ,
        "glass_loss":[] ,
        "head_loss":[] ,
        "points_loss":[] ,
        }

for  line in ctx:
    try:
        if 'Train'  in line:
            ele =  (line.split("All_loss")[1].split())
            Train_loss["All_loss"].append(float(ele[1]))
            Train_loss["gender_loss"].append(float(ele[3]))
            Train_loss["smile_loss"].append(float(ele[5]))
            Train_loss["glass_loss"].append(float(ele[7]))
            Train_loss["head_loss"].append(float(ele[9]))
            Train_loss["points_loss"].append(float(ele[11]))
        if 'Evalu' in line:
            ele =  (line.split("All_loss")[1].split())
            Evalu_loss["All_loss"].append(float(ele[1]))
            Evalu_loss["gender_loss"].append(float(ele[3]))
            Evalu_loss["smile_loss"].append(float(ele[5]))
            Evalu_loss["glass_loss"].append(float(ele[7]))
            Evalu_loss["head_loss"].append(float(ele[9]))
            Evalu_loss["points_loss"].append(float(ele[11]))
    except:
        print (line)
        quit()

def rm(ele):
    if ele>10:
        return 0
    else:
        return ele

def  draw(out_dir,cruve_type,x,y,x1=None,y1=None):
    # x is num of lst , y is value
    num = random.randint(0,5)
    color_lst = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(15,10))
    if x1 is None:
        plt.plot(x,y,label=cruve_type,color=color_lst[num],marker='o',linewidth=2)
    else:
        plt.plot(x,y,label=cruve_type.split('&')[0],color=color_lst[num],marker='o',linewidth=2)
        plt.plot(x1,y1,label=cruve_type.split('&')[1],color=color_lst[num+1],marker='o',linewidth=2)
    x_step = x[1]-x[0]
    if x_step >0:
        x_step = 30*x_step
    plt.xticks(np.arange(min(x), max(x)+1,x_step ))
    #plt.yticks(np.arange(0, max(y)))
    plt.legend()
    #plt.show()
    plt.savefig(out_dir+os.sep+cruve_type+'.png')


for i in range(len(kinds_lst)):
    Train_lst = Train_loss[kinds_lst[i]][start_loc:]
    Evalu_lst = Evalu_loss[kinds_lst[i]][start_loc:]

    x =[ele for ele in range(len(Train_lst))]
    if draw_flag :
        curve_type = 'Train_'+kinds_lst[i]+'&'+'Evalu_'+kinds_lst[i]
        train_curve_type = 'Train_'+kinds_lst[i]
        draw(out_dir,curve_type,x,Train_lst,x,Evalu_lst)
        draw(out_dir,train_curve_type,x,Train_lst)

#now the Evalu_lst is the last lst
min_loss = Evalu_lst[1]
for ele  in Evalu_lst:
    if ele < min_loss:
        min_loss = ele
print ("Epoch "+str(Evalu_lst.index(min_loss)) + " is the best Epoch. Point_Loss is "+str(min_loss))
if get_loss_flag:
    Train_lst = [str(ele) for ele in Train_loss[kinds_lst[1]][start_loc:]]
    Evalu_lst = [str(ele) for ele in Evalu_loss[kinds_lst[1]][start_loc:]]
    
    ctx = ' '.join(Train_lst) +'\n'
    ctx =ctx + ' '.join(Evalu_lst)
    file = open(get_loss_flag,'w')
    file.write(ctx)
    file.close()





