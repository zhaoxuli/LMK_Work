# -*- coding: utf-8 -*-
import symbol
import keras as ks
import numpy as np
import os
import keras.backend  as K
import warnings
import itchat
warnings.filterwarnings("ignore")
def run():
    # wechat setting
    itchat.auto_login(hotReload=True)
    users = itchat.search_friends(name="zhaoxuli")
    User_name = users[0]["UserName"]
    #data_setting
    img_h = 32
    img_w = 32
    img_c = 3
    epoch_num  =500
    Batch_size = 32
    batch_show = 10
    data_path = './data/train.npy'
    models_path = './cg_lr_points_models'
    model_name = 'cg_lr_points_model'
    log_file = './cg_lr_points_train.log'
    train_rate,evalu_rate = [0.9,0.1]
    learn_rate = 0.001

    def add_log(out_log,log_file):
        out_log = out_log+'\n'
        ctx = open(log_file,'a')
        ctx.write(out_log)
        ctx.close()

    if os.path.exists(log_file):
        print ("Log File is existed ")
        quit()
    if os.path.exists(models_path) ==False:
        os.makedirs(models_path)

    # 3094 = 32*32*3+22 = 3072+22
    # get train&label
    data_all = np.load(data_path)
    All_num = len(data_all)
    Batch_num = int(All_num/Batch_size)

    train_samples = int(All_num*train_rate)
    evalu_samples = int(All_num*evalu_rate)

    train_data = data_all[:train_samples]
    evalu_data = data_all[-evalu_samples:]

    [x_train,y_train] =[train_data[:,:3072]/255,train_data[:,-22:]]
    [x_evalu,y_evalu] =[evalu_data[:,:3072]/255,evalu_data[:,-22:]]

    x_train = x_train.reshape((train_samples,32,32,3))
    x_evalu = x_evalu.reshape((evalu_samples,32,32,3))

    points_train = y_train[:,-22:-4]
    points_evalu = y_evalu[:,-22:-4]
    cls_train  =[y_train[:,i] for i in range(-4,0)]
    cls_evalu = [y_evalu[:,i] for i in range(-4,0)]

    [gender_cls_train,smile_cls_train,glassese_cls_train,head_pose_cls_train] = [ks.utils.np_utils.to_categorical(ele) for ele in cls_train]
    [gender_cls_evalu,smile_cls_evalu,glassese_cls_evalu,head_pose_cls_evalu] = [ks.utils.np_utils.to_categorical(ele) for ele in cls_evalu]

    label_train_lst = [gender_cls_train,smile_cls_train,glassese_cls_train,head_pose_cls_train,points_train]
    label_evalu_lst = [gender_cls_evalu,smile_cls_evalu,glassese_cls_evalu,head_pose_cls_evalu,points_evalu]

    #compile model and multi_loss
    model = symbol.build((img_h,img_w,3))

    output_lst = [ "gender_output","smile_output","glassese_output","head_output","points_output"]
    losses = {
       "gender_output":"categorical_crossentropy",
       "smile_output":"categorical_crossentropy",
       "glassese_output":"categorical_crossentropy",
       "head_output":"categorical_crossentropy",
       "points_output":"mse"
    }
    lossWeights = {
        "gender_output":0,
        "smile_output":0,
        "glassese_output":0,
        "head_output":0,
        "points_output":10
     }
    adam =ks.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer = adam,loss = losses,loss_weights = lossWeights)#metrics = ["acc"])

    def Eearly_stopping(train_loss_lst,evalu_loss_lst,eata,loss_weight):
        k  = len(train_loss_lst)


    # train
    try:
        for epoch in range(epoch_num):
            print ('\n----------------Epoch '+str(epoch)+' ------------------------')
            if (epoch%50 ==0) and (epoch!=0):
                learn_rate = learn_rate*0.8
                out_log =  (("[Change lr in Epoch %d to %f]")%(epoch ,learn_rate))
                add_log(out_log,log_file)
                print (out_log)
                adam =ks.optimizers.Adam(lr=learn_rate)
                model.compile(optimizer = adam,loss = losses,loss_weights = lossWeights)
            for k in range(Batch_num):
                #get batch data
                batch_x_train = x_train[k*Batch_size:Batch_size*(k+1)]
                batch_label_train = [label_ele[k*Batch_size:Batch_size*(k+1)] for label_ele in  label_train_lst]
                #check batch size is not null
                if batch_x_train.shape[0] == Batch_size:
                    train_out = model.train_on_batch(batch_x_train,batch_label_train)
                    # cacul the epoch_loss
                    if (epoch ==0) and (k==0):
                        epoch_loss = train_out
                    else:
                        out_len = len(train_out)
                        for i in range(out_len):
                            epoch_loss[i] = epoch_loss[i] + train_out[i]
                    #show information  and save log
                    if (k%batch_show == 0):
                        print ("All_loss: %0.3f gender_loss: %0.3f smile_loss: %0.3f glass_loss: %0.3f head_loss: %0.3f points_loss: %04f" \
                                % (train_out[0],train_out[1],train_out[2],train_out[3],train_out[4],train_out[5]))

            epoch_loss = [float(ele)/float(Batch_num) for ele in epoch_loss]
            out_log =  ("\n[Train Epoch %d] All_loss: %0.3f gender_loss: %0.3f smile_loss: %0.3f glass_loss: %0.3f head_loss: %0.3f points_loss: %04f" \
                           % (epoch,epoch_loss[0],epoch_loss[1],epoch_loss[2],epoch_loss[3],epoch_loss[4],epoch_loss[5]))
            print (out_log)
            add_log(out_log[1:],log_file)

            evalu_loss = model.test_on_batch(x_evalu,label_evalu_lst)
            out_log =  ("[Evalu Epoch %d] All_loss: %0.3f gender_loss: %0.3f smile_loss: %0.3f glass_loss: %0.3f head_loss: %0.3f points_loss: %0.4f" \
                           % (epoch,evalu_loss[0],evalu_loss[1],evalu_loss[2],evalu_loss[3],evalu_loss[4],evalu_loss[5]))
            print (out_log)
            add_log(out_log,log_file)
            # save model
            filepath =models_path+os.sep+model_name+'-'+str(epoch) +'-'+("%0.3f")%(train_out[-1]*10) +'.hdf5'
            model.save(filepath, overwrite=True)


        flag = itchat.send(msg ='You have done trainning',toUserName=User_name)
        print (flag)
    except:
        flag = itchat.send(msg ='Failed to train',toUserName=User_name)
        print (flag)
    #os.system("shutdown -s -t 300")
if __name__ =='__main__':
    run()
