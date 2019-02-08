# coding:utf-8
import numpy as np
import tensorflow as tf
import codecs
import re
# 影评数据文件路径
neg_filepath = 'neg_data.txt'
pos_filepath = 'pos_data.txt'

neg_idsMatrix = './data/neg_idsMatrix.npy'
pos_idsMatrix = './data/pos_idsMatrix.npy'
pos_data = []
pos_label = []
neg_data = []
neg_label = []
max_word_length = 250

# 存储词向量
neg_Mat = []
pos_Mat = []


wordList = np.load("./npy/wordsList.npy")
print("loading succeed")
# 转换成列表
wordList = wordList.tolist()
# utf-8编码
wordList = [w.decode('utf-8') for w in wordList]

# wordVectors = np.load("./npy/wordVectors.npy")
def loaddata(neg_file, pos_file):
    global pos_data, neg_data,neg_label,pos_label
    # 加载消极影评
    with codecs.open(neg_file, 'r',' utf-8') as file:
        neg_data = file.readlines()
        neg_data = neg_data[:12500]
        neg_label = [0 for i in range(len(neg_data))]
    # 加载积极影评
    with codecs.open(pos_file, 'r', ' utf-8') as file:
        pos_data = file.readlines()
        pos_data = pos_data[:12500]
        pos_label = [1 for i in range(len(pos_data))]




def saveIdsMatrix(neg, file):
    # with open(idsMatrix,'w',encoding='utf-8') as f:
    #     for i in neg:
    #         a = []
    #         pattern = re.compile(r"[*;().,0-9?!'\"\\/<br>]")
    #         # 去掉特殊字符， 去掉首位空格，并分割
    #
    #         wordMax = re.sub(pattern,'',i).strip("\r\n").split()
    #         print(wordMax)
    #         for j in wordMax:
    #             try:
    #                 a.append(wordList.index(j.lower()))
    #             except ValueError:
    #                 a.append(399999)
    #     # 判断词向量是否小于250 维度 小于则填补0 打于 则去掉多余
    #
    #         if len(a) < max_word_length:
    #             data = [0] * (250 - len(a))
    #             a.extend(data)
    #         else:
    #             a = a[:250]
    #         f.write(str(a))
    arry = []
    for i in neg:
        a = []
        pattern = re.compile(r"[*;().,0-9?!'\"\\/<br>]")
        # 去掉特殊字符， 去掉首位空格，并分割

        wordMax = re.sub(pattern,'',i).strip("\r\n").split()
        # print(wordMax)
        for j in wordMax   :
            try:
                a.append(wordList.index(j.lower()))
            except ValueError:
                a.append(399999)
    # 判断词向量是否小于250 维度 小于则填补0 打于 则去掉多余

        if len(a) < max_word_length:
            data = [0] * (250 - len(a))
            a.extend(data)
        else:
            a = a[:250]
        # np.save(file, a)
        if a[0] != 0:
            print(a)
            arry.append(a)
    arry = np.array(arry)
    print("开始写入")
    np.save(file, arry)

loaddata(neg_filepath, pos_filepath)
print("加载成功")
# print(pos_data[0])
saveIdsMatrix(neg_data, neg_idsMatrix)

saveIdsMatrix(pos_data, pos_idsMatrix)
print("完成")


