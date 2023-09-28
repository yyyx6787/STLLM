
import pandas as pd
import pickle
# from shapely.geometry import Point, LineString
# from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn
import networkx as nx
import numpy as np


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

# test_vector = load_data("./zemob.pickle")
# print(test_vector.size())
# println()
region = load_data("../data/param/hy_vector_signal_trans_sc_1.pickle")
spatial_vector = load_data("../data/region_spatial_refine.pickle") 

hy = load_data("../data/hy_new_10.pickle") 
# hy = load_data("../data/hy_new_test_80.pickle") 
reg_flow = load_data("../data/reg_vector_dict.pickle")
com_flow = load_data("../data/com_flow_spatial.pickle")

# print("reg_flow:", len(reg_flow))
# print(type(spatial_vector))
# print(type(reg_flow))
# com_flow_spatial ={}
# for ii,uu in zip(spatial_vector.items(),reg_flow.items()):
#     if ii[0]==uu[0]:
#         cc = np.mean((torch.squeeze(ii[1],0).detach().numpy().tolist(), uu[1]), axis = 0)
#         com_flow_spatial[ii[0]]=cc.tolist()
# print(len(com_flow_spatial))
# file=open(r"../data/com_flow_spatial.pickle","wb")
# pickle.dump(com_flow_spatial,file) #storing_list
# file.close()        
# println()
# print(region.size())


# print(check_vector[0].size())
# print(hy.nodes())
# print(hy)
# println()
nn_1 = nn.Linear(192,96)
hy_nodes_dict={}
for n,n_vec in zip(hy.nodes(),region):
    tp = n.split("_")[1]
    # print("tp:", tp)
    # print(n)
    if tp not in hy_nodes_dict.keys():
        hy_nodes_dict[int(tp)] = []
        hy_nodes_dict[int(tp)].append(n_vec.tolist())
    else:
        hy_nodes_dict[int(tp)].append(n_vec.tolist())
# print(":",len(hy_nodes_dict))
# print(hy_nodes_dict.keys())
# println()
for i in range(180):
    if i not in hy_nodes_dict.keys():
        print("--here--")
        hy_nodes_dict[i] = [torch.squeeze(spatial_vector[i],0).tolist()]

hy_com  = {}
for key,value in hy_nodes_dict.items():
    tmp = np.mean(value, axis=0).tolist()
    # print(tmp)
    tmp_ = torch.tensor(tmp).tolist()
    hy_com[int(key)]  = tmp_


linear = nn.Linear(96, 16)
nyc_vec = []
for key,value in hy_com.items():
    nyc_vec.append(value)
hycom_vec = []
for key,value in hy_com.items():
    hycom_vec.append(linear(torch.tensor(value)).tolist())
vec_final_ = np.reshape(np.tile(np.array(hycom_vec),(30,4)),(180,30,4,16))

print("hycom_vec", torch.tensor(np.array(nyc_vec)).size())
# println()
file=open(r"../data/param_pkl/vector_nyc.pickle","wb")
pickle.dump( torch.tensor(np.array(nyc_vec)),file) #storing_list
file.close()
print("---finish---")



println()
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
import mglearn

# 读取数据，并划分训练集和测试集
X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
# 通过设置不同的alpha值建立三个lasso实例
lasso = Lasso().fit(X_train,y_train)
lasso001 =Lasso(alpha=0.01).fit(X_train,y_train)
lasso00001 = Lasso(alpha=0.0001).fit(X_train,y_train)
print('**********************************')
print("Lasso alpha=1")
print ("training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))

print('**********************************')
print("Lasso alpha=0.01")
print ("training set score:{:.2f}".format(lasso001.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso001.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso001.coef_!=0)))

print('**********************************')
print("Lasso alpha=0.0001")
print ("training set score:{:.2f}".format(lasso00001.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso00001.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso00001.coef_!=0)))


# check = {}
# for key,value in check_vector.items():
# 	v = nl(value)
# 	print(key, v.size())














