# -*- coding: utf-8 -*-

import itchat 

def run():
    itchat.auto_login(hotReload=True)
    users = itchat.search_friends(name="zhaoxuli")
    #print (users)
    User_name = users[0]["UserName"]
    flag = itchat.send(msg ='You have done trainning',toUserName=User_name)
    print (flag)
