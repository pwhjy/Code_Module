# -*- coding: utf-8 -*-
from urllib import parse
import base64
import hashlib
import time
import requests
import json
# OCR手写文字识别接口地址
URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/handwriting"
# 应用APPID(必须为webapi类型应用,并开通手写文字识别服务,参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
APPID = "737c5cad"
# 接口密钥(webapi类型应用开通手写文字识别后，控制台--我的应用---手写文字识别---相应服务的apikey)
API_KEY = "881a3b69f1e40b5693a853417f5e5141"

def getHeader():
    curTime = str(int(time.time()))
    param = "{\"language\":\""+language+"\",\"location\":\""+location+"\"}"
    paramBase64 = base64.b64encode(param.encode('utf-8'))

    m2 = hashlib.md5()
    str1 = API_KEY + curTime + str(paramBase64, 'utf-8')
    m2.update(str1.encode('utf-8'))
    checkSum = m2.hexdigest()
	# 组装http请求头
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    return header
def getBody(filepath):
    with open(filepath, 'rb') as f:
        imgfile = f.read()
    data = {'image': str(base64.b64encode(imgfile), 'utf-8')}
    return data
# 语种设置
language = "cn|en"
# 是否返回文本位置信息
location = "false"
# 图片上传接口地址
picFilePath = "./12.jpeg"
# headers=getHeader(language, location)
r = requests.post(URL, headers=getHeader(), data=getBody(picFilePath))
#print(r.content)

def sstr(r):
    new_s = r.content.decode()
    d = json.loads(new_s)
    for lines in d["data"]["block"]:
        for line in lines['line']:
            for str in line["word"]:
                print(str["content"])
#sstr(r)

def ocr_pic(picFilePath):
    all_strings = []
    r = requests.post(URL, headers=getHeader(), data=getBody(picFilePath))
    new_s = r.content.decode()
    #print(new_s)
    d = json.loads(new_s)
    for lines in d["data"]["block"]:
        for line in lines['line']:
            for str in line["word"]:
                all_strings.append(str["content"])
    result = ''.join(all_strings)
    return result


print(ocr_pic(picFilePath))