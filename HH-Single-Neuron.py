import numpy as np
import matplotlib.pyplot as plt

class HH_neuron:
    def __init__(self,
                 Cm = 1.0, # microFarad
                 ENa = 50,   # miliVolt
                 EK = -77,   # miliVolt
                 El = -54,   # miliVolt
                 g_Na = 120, # mScm-2
                 g_K = 36,   # mScm-2
                 g_l = 0.03, # mScm-2
                 dt = 0.05,
                 init_V = -60,
                 ) -> None:
        self.Cm = Cm
        self.ENa = ENa
        self.EK = EK
        self.El = El
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_l = g_l
        self.dt = dt
        self.init_V = init_V
        self.reset()
        
    def reset(self, ):
        
        #Initialize the voltage and the channels :
        self.v, self.m, self.h, self.n = [], [], [], []
        self.v.append(self.init_V)
        self.m.append(self.alphaM(self.v[0])/(self.alphaM(self.v[0])+self.betaM(self.v[0])))
        self.n.append(self.alphaN(self.v[0])/(self.alphaN(self.v[0])+self.betaN(self.v[0])))
        self.h.append(self.alphaH(self.v[0])/(self.alphaH(self.v[0])+self.betaH(self.v[0])))
        self.i = 1
        self.tt = []; self.tt.append(0)
    
    # define alhpa constant and betha constant for n,m,h channels for opening probability
    def alphaN(self, v): return 0.01*(v+50)/(1-np.exp(-(v+50)/10))
    def betaN(self, v):  return 0.125*np.exp(-(v+60)/80)
    def alphaM(self, v): return 0.1*(v+35)/(1-np.exp(-(v+35)/10))
    def betaM(self, v):  return 4.0*np.exp(-0.0556*(v+60))
    def alphaH(self, v): return 0.07*np.exp(-0.05*(v+60))
    def betaH(self, v):  return 1/(1+np.exp(-(0.1)*(v+30)))

    def step(self, I):
        '''
        I: current in time=i
        '''
        dt = self.dt
        # solving ODE using Euler's method:
        self.m.append(self.m[self.i-1] + dt*((self.alphaM(self.v[self.i-1])*(1-self.m[self.i-1]))-self.betaM(self.v[self.i-1])*self.m[self.i-1]))
        self.n.append(self.n[self.i-1] + dt*((self.alphaN(self.v[self.i-1])*(1-self.n[self.i-1]))-self.betaN(self.v[self.i-1])*self.n[self.i-1]))
        self.h.append(self.h[self.i-1] + dt*((self.alphaH(self.v[self.i-1])*(1-self.h[self.i-1]))-self.betaH(self.v[self.i-1])*self.h[self.i-1]))
        gNa = self.g_Na * self.h[self.i-1] * (self.m[self.i-1])**3
        gK = self.g_K * self.n[self.i-1]**4
        gl = self.g_l
        INa = gNa * (self.v[self.i - 1] - self.ENa)
        IK = gK * (self.v[self.i - 1] - self.EK)
        Il = gl * (self.v[self.i - 1] - self.El)
        #if you want to use some dynamic current try to uncomment the below line and comment the line aftar that.
        #self.v.append(v[i-1]+(dt)*((1/Cm)*(I[i-1]-(INa+IK+Il))))
        self.v.append(self.v[self.i-1]+self.dt*((1/self.Cm)*(I-(INa+IK+Il))))
        self.tt.append(self.tt[-1]+dt)
        self.i += 1

if __name__ == '__main__':
    hh = HH_neuron()
    # [hh.step(6.8) for i in range(10000)]
    for i in range(1000):
        if i < 200:
            hh.step(7)
        else:
            hh.step(0)

#Plot the data
# plt.figure(figsize=(15,20))
# plt.legend(loc='upper left')
# plt.title('Hodgkin Huxely Spike Model')
# plt.subplot(3,1,1)
# plt.plot(hh.tt, hh.v,'b-',label='voltage')
# plt.legend(loc='upper left')
# plt.xlabel('time (ms)')
# plt.ylabel('Voltage')

# plt.subplot(3,1,2)
# plt.plot(hh.tt, hh.n,'y-',label='n channels')
# plt.legend(loc='upper left')
# plt.xlabel('time')
# plt.ylabel('n')
# plt.subplot(3,1,2)
# plt.plot(hh.tt, hh.m,'g-',label='m channels')
# plt.legend(loc='upper left')
# plt.xlabel('time')
# plt.ylabel('m')
# plt.subplot(3,1,2)
# plt.plot(hh.tt, hh.h,'r--',label='h channels')
# plt.legend(loc='upper left')
# plt.xlabel('time')
# plt.ylabel('channels')

# plt.subplot(3,1,3)
# plt.plot(hh.n, hh.v,'r--',label='Phase plane')
# plt.legend(loc='upper left')
# plt.xlabel('voltage')
# plt.ylabel('n channels')
# plt.show()