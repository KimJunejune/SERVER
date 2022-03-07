# -----------------------------
# 本py于SaveAsTxt.py后执行
#   1. 读取保存到txt文件中的日志, 故障信息等数据
#   2. 将日志信息进行分词
#   3. 用Doc2Vec生成段落向量, 用作train_features
#   4. 将train_features和train_labels保存为npy格式
#
# train_features和train_labels都是16999维
# ------------------------------
import os
import nltk
# nltk.download('punkt')
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier as rf
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import warnings
warnings.filterwarnings('ignore')

data_dir = './'
# --------------------------------
# 加载服务器名，日志等：
#     sn_list: 长13705的 服务器名列表
#     tail_msg_list : 长13705的 以字符串格式存储的 13705台服务器的日志信息
#               ' Drive Slot HDD_L_14_Status | Drive Fault | Asserted. Drive Slot / Bay HDD_L_14_Status | Drive Fault | Asserted. Drive Slot HDD_L_14_Status | Drive Fault | Deasserted. Drive Slot / Bay HDD_L_14_Status | Drive Fault | Deasserted',
#     tokenized_sent : 长137005的 以列表格式存储的 每条日志的分词
#               ['drive','slot','hdd_l_14_status', '|','drive','fault','|','asserted','.','drive','slot','/','bay','hdd_l_14_status','|','drive','fault','|','asserted','.','drive','slot','hdd_l_14_status','|','drive','fault','|','deasserted','.','drive','slot','/','bay','hdd_l_14_status','|','drive','fault','|','deasserted'],
# --------------------------------

# 读取服务器列表
sn_list = []
with open(os.path.join(data_dir, "sn_list.txt"), "r", encoding= "utf-8") as f:
    for line in f.readlines():
        sn_list.append(line.strip())
# 读取日志列表
msg_list = []
with open(os.path.join(data_dir, "msg_list.txt"), "r", encoding="utf-8") as f:
    for line in f.readlines():
        msg_list.append(line.strip())
# 加载每条日志的分词
tokenized_sent = [word_tokenize(s.lower()) for s in msg_list]

# --------------------------------
# embedding前的工作： 使用index顺序标记词向量
# tagged_data:
#     长13705的 以列表格式存储的 进行标记后的tokenized_sent
'''
tagged_data = [
    TageedDocument(tokenized_data[0], [0]), 
    TageedDocument(tokenized_data[1], [1]), 
    TageedDocument(tokenized_data[2], [2]), 
]
'''
# --------------------------------
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]

# -------------------------------
# Doc2Vec模型：
#   模型是基于Word2Vec基础上，引入了段落的概念
#   Word2Vec将每个单词用一个唯一词向量进行表示
#   Doc2Vec则是将词向量扩充成段落向量，
#   所以Doc2Vec模型需要的输入格式就是TaggedDocument： (词列表， 段落序号)
#
# 模型参数：
#   - sentences: 需要TaggedDocument格式的输入
#   - alpha: 学习率
#   - size: 特征向量的维度 默认100
#   - window: 表示当前词 和 预测词 在一个句子中最大距离是多少
#   - min_count: 词频少于min_count的单词会被丢弃, 默认为5
#
# 更多参数参考： https://blog.csdn.net/mpk_no1/article/details/72510655?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164662087516780265422133%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164662087516780265422133&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-72510655.pc_search_result_control_group&utm_term=doc2vec%E5%8F%82%E6%95%B0&spm=1018.2226.3001.4187
# 模型数学理论： https://blog.csdn.net/itplus/article/details/37969635
# -------------------------------
model = Doc2Vec(tagged_data, vector_size = 10, window = 2, min_count = 1, epochs = 10)

# -------------------------------------
# 读取log_label_list.txt文件
# 将log保存到raw_train中
# 将lables保存到train_lable中
# -------------------------------------
raw_train = []
train_lable = []

with open('./log_label_list.txt', "r", encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        content = line.split('$')
        raw_train.append(content[0])
        train_lable.append(int(content[1]))
train_tokenized = [word_tokenize(s) for s in raw_train]
# --------------------------------
# 创建训练集
# 将raw_train中的字符串转换成词向量，用shuzubaocun
# model.infer_vector()
#   - doc_words: 字符串 或 列表
#   - alpha: 学习率
#   - epochs
# infer_vector()根据 model的输入TaggedDocument构建一个模型
# 对于传入infer_vector()的分词列表创建一个推断词向量
# ---------------------------------
train_data = []
for i in range(len(train_lable)):
    train_data.append(model.infer_vector(train_tokenized[i]))

train_features = np.array(train_data)
train_label = np.array(train_lable)

np.save('./processed/train_features.npy', train_features)
np.save('./processed/train_labels.npy', train_label)