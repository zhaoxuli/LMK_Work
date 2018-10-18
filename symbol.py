# -*- coding: utf-8 -*-

import keras as ks

def conv(inputs,num_filters=16,kernel_size=3,strides=1,ACT='relu',BN=True):

    x = ks.layers.Conv2D(num_filters,kernel_size=kernel_size,strides=strides,
                         padding='same',kernel_initializer='he_normal',
                         kernel_regularizer=ks.regularizers.l2(1e-4))(inputs)
    if BN:
        x = ks.layers.BatchNormalization()(x)
    if ACT is not False:
        x = ks.layers.Activation(activation='relu')(x)
    return x

def cls_out_layer(x,out_num = 2,cls_name="default"):
    x = conv(x,64,1,ACT=False)
    #x = ks.layers.GlobalMaxPool2D()(x)
    x = ks.layers.AveragePooling2D(pool_size=4)(x)
    x = ks.layers.Flatten()(x)
    cls = ks.layers.Dense(out_num,activation='softmax',name = cls_name)(x)
    return cls

def reg_out_layer(x,out_num = 18,reg_name="default"):
    x = conv(x,64,1,ACT=False)
    #x = ks.layers.GlobalMaxPool2D()(x)
    x = ks.layers.AveragePooling2D(pool_size=4)(x)
    x = ks.layers.Flatten()(x)
    reg = ks.layers.Dense(out_num, name = reg_name)(x)
    return reg


def  Network(input_shape):
    #input_shape is 32
    # gender  smile  galsses  head pose
    x = input_shape
    x = conv(x,32,3)
    x = conv(x,32,3)
    x = ks.layers.MaxPool2D(pool_size=(2,2),padding="same")(x)

    x = conv(x,64,3)
    x = conv(x,64,3)
    x = ks.layers.MaxPool2D(pool_size=(2,2),padding="same")(x)

    x = conv(x,128,3)
    x = conv(x,128,3)
    x = ks.layers.MaxPool2D(pool_size=(2,2),padding="same")(x)

    gender_cls = cls_out_layer(x,cls_name = "gender_output")
    smile_cls = cls_out_layer(x,cls_name = "smile_output")
    glasses_cls = cls_out_layer(x,cls_name = "glassese_output")
    pose_cls = cls_out_layer(x,cls_name="head_output",out_num=5)
    points_reg = reg_out_layer(x,reg_name="points_output")

    return  gender_cls,smile_cls,glasses_cls,pose_cls,points_reg


def  build(input_shape):
    inputs = ks.layers.Input(shape = input_shape)
    gender_cls,smile_cls,glasses_cls,pose_cls,points_reg = Network(inputs)

    model = ks.Model(
        inputs = inputs,
        outputs = [gender_cls,smile_cls,glasses_cls,pose_cls,points_reg],
        name = 'TCDCN_net'
    )

    return  model
