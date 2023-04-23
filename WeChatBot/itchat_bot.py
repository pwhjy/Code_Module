import itchat
import os
import requests
# from PIL import Image
# import numpy as np
from itchat.content import *
from ocr import ocr_pic

@itchat.msg_register(TEXT)
def text_reply(msg):
    # 监听文本消息
    friend = itchat.search_friends(userName=msg['FromUserName'])
    if msg['Content'] == '我需要手写字体识别':
        itchat.send_msg('请发送给我一张图片', toUserName=msg['FromUserName'])

    @itchat.msg_register([PICTURE])
    def download_files(msg):
        # 监听图片消息
        if msg['FromUserName'] == friend['UserName']:
            img_file = msg['FileName']
            msg.download(img_file)
            # img = Image.open(img_file)
            # img_arr = np.array(img)
            # h, w = img_arr.shape[:2]
            str = ocr_pic(img_file)
            
            itchat.send_msg(str, toUserName=msg['FromUserName'])
            #itchat.send_msg(f'您发来的图片大小为 {h} * {w} 像素', toUserName=msg['FromUserName'])
            # 删除文件
            os.remove(img_file)

if __name__ == '__main__':
    itchat.auto_login(hotReload=False)
    itchat.run()
