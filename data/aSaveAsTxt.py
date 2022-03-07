# -------------------------------------------
# 本py实现：
# 1. 将日志的csv文件保存为txt，用列表存储所有日志信息和服务器名称
# 2. 生成 日志信息 对应唯一 label， 并另存到log_label.txt中
#
# 日志信息: 一个SERVER的所有日志信息 -->
# 服务器名称:
# 训练集: (故障发生前最后10条日志 + 故障类型)  作为一个样本，中间用'$'分隔
# -------------------------------------------
import os
import nltk
# -------------------------
# NLTK自然语言处理库
# 需要下载 'puntk' 文件
# 地址：http://www.nltk.org/nltk_data/
# 下载puntk model后解压到 /nltk_data/tokenizers
# -------------------------
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier as rf
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 读取sel日志，排序
data_dir = "./raw"
sel_data = pd.read_csv(os.path.join(data_dir, './preliminary_sel_log_dataset.csv'))
sel_data.sort_values(by=['sn', 'time'], inplace=True)
sel_data.reset_index(drop=True, inplace=True)

# --------------------------------
# 运行：
#     sn_list: 长13705的 服务器名列表
#     tail_msg_list : 长13705的 以字符串格式存储的 13705台服务器的日志信息   ' Drive Slot HDD_L_14_Status | Drive Fault | Asserted. Drive Slot / Bay HDD_L_14_Status | Drive Fault | Asserted. Drive Slot HDD_L_14_Status | Drive Fault | Deasserted. Drive Slot / Bay HDD_L_14_Status | Drive Fault | Deasserted',
# --------------------------------
sn_list = sel_data['sn'].drop_duplicates(keep='first').to_list()   # 统计所有SERVER服务器  ---->  共13705台服务器
tail_msg_list = ['.'.join(sel_data[sel_data['sn']==i]['msg'].tail(10).to_list()) for i in sn_list]  # 取出每台服务器的最后十条日志， 同一台服务器的日志信息用.连接，保存为字符串格式

# --------------------------------
# 保存：
# 将两个列表保存为txt格式，方便下次读取
# 至此，服务器名 和 日志信息另存完毕
# --------------------------------
with open("./sn_list.txt", 'w', encoding='utf-8')  as f:
    for sn in sn_list:
        f.writelines(sn)
        f.writelines("\n")  # 须分行

with open("./msg_list.txt", 'w', encoding='utf-8')  as f:
    for msg in tail_msg_list:
        f.writelines(msg)
        f.writelines("\n")

# --------------------------------
# 另存标签df
# label中的SERVER可能会出现重复的情况
# 重复表明 同一个SERVER在不同时间出现多次不同LABEL的故障
#
# 将故障发生前的最后十条信息以txt格式保存
# 每行由 日志信息+故障标签 组成
# 日志信息和故障标签用'$'保存——已确认日志信息中无'$'
# --------------------------------
label = pd.read_csv(os.path.join(data_dir, 'raw/preliminary_train_label_dataset.csv'))
label.sort_values(by=['sn', 'fault_time'], inplace=True)
label.reset_index(drop=True, inplace=True)

# 保存每条故障发生前10条日志，用字符串格式存储
label_list = []
for i, row in label.iterrows():
    label_list.append('.'.join(sel_data[(sel_data['sn']==row['sn'])&(sel_data['time']<=row['fault_time'])].tail(10)['msg']).lower())
train_label = label['label'].values

with open("./log_label_list.txt", 'w', encoding='utf-8') as f:
    for i in range(len(label_list)):
        log_label = label_list[i] + "$" + str(train_label[i])
        f.writelines(log_label)
        f.writelines('\n')

