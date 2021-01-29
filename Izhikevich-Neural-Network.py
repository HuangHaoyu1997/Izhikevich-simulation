# -*- coding: utf-8 -*-
"""
Created on 2021年1月29日19:40
@author: Haoyu Huang
Python 3.6.2
"""

import numpy as np
import matplotlib.pyplot as plt
import time

Ne = 800 # 兴奋性神经元，4:1
Ni = 200 # 抑制性神经元
thr = 30 # 发放阈值
time_span = 500 # 仿真步长
re = np.random.rand(Ne,1) 
ri = np.random.rand(Ni,1)

a = np.concatenate((0.02*np.ones((Ne,1)),0.02+0.08*ri)) # 通过随机数生成参数不同的1000个神经元
b = np.concatenate((0.2*np.ones((Ne,1)),0.25-0.05*ri))
c = np.concatenate((-65+15*re**2,-65*np.ones((Ni,1))))
d = np.concatenate((8-6*re**2,2*np.ones((Ni,1))))
S = np.concatenate((0.5*np.random.rand(Ne+Ni,Ne),-np.random.rand(Ne+Ni,Ni)),1) # 每个Neuron都与其余999个相连，但兴奋性Neuron的权重>0,抑制性Neuron权重<0

v = -65*np.ones((Ne+Ni,1)) # 动力学方程的初始状态
u = b*v
firings = np.array([[0,0]])
for t in range(time_span):
    I = np.concatenate((5*np.random.randn(Ne,1),2*np.random.randn(Ni,1)),0) # thalamic input，模拟丘脑的噪声输入
    fired = np.where(v>=thr)[0] # 发放脉冲神经元的index
    if fired != []:
        firing_idx = np.concatenate((t*np.ones((len(fired),1)),fired.reshape(-1,1)),1) # firing_idx第一列表示time_step，第二列代表Firing Neuron Index
        firings = np.concatenate((firings,firing_idx))
        v[fired] = c[fired] # update
        u[fired] = u[fired] + d[fired]
        I += S[:,fired].sum(1).reshape(-1,1)
    v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I) # 原论文表示以0.5ms为单位进行模拟比较稳定
    v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)
    u = u + a*(b*v - u)

plt.scatter(firings[:,0],firings[:,1],s=1)
plt.xlabel('time/ms')
plt.ylabel('neuron index')
plt.savefig('./images/Izhikevich-Network.png',dpi=500)





