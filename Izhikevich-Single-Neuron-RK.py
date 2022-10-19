# -*- coding: utf-8 -*-
"""
Created on 2021年1月29日19:40
@author: Haoyu Huang
Python 3.6.2
"""
import numpy as np
import matplotlib.pyplot as plt
import time

a = 0.15
b = 0.25
c = -65
d = 2
v_th = 30 # 发放阈值
h = 1 # 仿真步长
v1 = np.array([c,b*c]) # vu初始值
amplitude = 1 # 直流电强度
method = 'RK1' # 数值模拟方法
sim_time = 500 # 仿真时间

II = amplitude*np.ones((sim_time),dtype=np.float) # 直流输入
II[0: 100] = 0 # 前100ms电流为0

def function(value,I):
    '''
    求解函数
    :param value: 函数变量
    :param I: 电流输入
    :return: 求解的结果
    '''
    d1 = 0.04*value[0]**2 + 5*value[0] + 140 - value[1] + I
    d2 = a*(b*value[0] - value[1])
    return np.array(d1),np.array(d2)

def runge_ketta(f, h, y_in, I, method='RK41'):
    '''
    龙格库塔法求解Izhikevich模型,所有的数据为np.array类型
    :param f:函数
    :param h:步长
    :param y_in:初始值
    :param I: 电流输入
    :return:函数的结果
    '''
    h = np.array(h)
    if method == 'RK4':
        alfa = np.array([1,2,2,1])*(1/6)
        theta = np.array([0,0.5,0.5,1])
        k1 = h*f(y_in,I)
        k2 = h*f(y_in + theta[1]*k1,I)
        k3 = h*f(y_in + theta[2]*k2,I)
        k4 = h*f(y_in + k3,I)
        v = y_in + alfa[0]*k1 + alfa[1]*k2 + alfa[2]*k3 + alfa[3]*k4
    if method == 'RK3':
        alfa = np.array([1,4,1])*(1/6)
        k1 = h*f(y_in,I)
        k2 = h*f(y_in + 0.5*k1,I)
        k3 = h*f(y_in - k1 + 2*k2,I)
        v = y_in + alfa[0]*k1 + alfa[1]*k2 + alfa[2]*k3
    if method == 'RK2':
        alfa = np.array([0.5,0.5])
        k1 = h*f(y_in,I)
        k2 = h*f(y_in + k1,I)
        v = y_in + alfa[0]*k1 + alfa[1]*k2
    if method == 'RK1':
        k1 = h*f(y_in,I)
        v = y_in + k1
    return v

d11 = [] # list for v
d12 = [] # list for u
d11.append(v1[0])
d12.append(v1[1])

start_time = time.time()
for I in II:
    v1 = runge_ketta(function,h,v1,I,method)
    if v1[0] > v_th:
        d11.append(v_th) # firing
        v1[0] = c        # reset
        v1[1] = v1[1] + d
    else:
        d11.append(v1[0])
    d12.append(v1[1])
end_time = time.time()

tspan = [i for i in np.arange(0,5,0.01)] # 可以变换时间尺度

plt.figure(1)
plt.plot(tspan,d11[0:500]) # v
plt.plot(tspan,d12[0:500]) # u 
plt.plot(tspan,II[0:500])
plt.grid()
plt.legend(['v','u','I'])
plt.xlabel('time / s')
plt.ylabel('membrane potential / V')
plt.savefig('Izhikevich-Single-Neuron.png',dpi=500)

print(method,1000*(end_time-start_time),'ms')




