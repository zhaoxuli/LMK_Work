# -*- coding: utf-8 -*-

import sgd_equal_run
import sgd_singal_run
import shuffle_sgd_equal_run

import itchat

itchat.auto_login(hotReload=True)

users = itchat.search_friends(name="zhaoxuli")
#print (users)
User_name = users[0]["UserName"]

flag = itchat.send(msg ='Start Trainning ',toUserName=User_name)
try:
    sgd_equal_run.run(False)
    flag = itchat.send(msg ='You have done trainning',toUserName=User_name)
    print (flag)
    sgd_singal_run.run(False)
    flag = itchat.send(msg ='You have done trainning',toUserName=User_name)
    print (flag)
    shuffle_sgd_equal_run.run(False)
    flag = itchat.send(msg ='You have done trainning',toUserName=User_name)
    print (flag)

except:
    flag = itchat.send(msg ='Trainning Failed',toUserName=User_name)
    print (flag)
