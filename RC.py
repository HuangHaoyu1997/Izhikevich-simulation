import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.distributions import Poisson, Uniform
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class RC:
    def __init__(self,
                 N_input,
                 N_hidden,
                 N_output,
                 alpha,
                 decay,
                 threshold,
                 
                 ) -> None:
        self.N_in = N_input
        self.N_hid = N_hidden
        self.N_out = N_output
        self.alpha = alpha
        self.decay = decay
        self.thr = threshold
        self.reset()
        
    def reset(self,):
        self.W_in = np.random.uniform(low=np.zeros((self.N_hid, self.N_in)), 
                                      high=np.ones((self.N_hid, self.N_in))*0.1)
        self.A = np.random.uniform(low=-1*np.ones((self.N_hid, self.N_hid)), 
                                   high=np.ones((self.N_hid, self.N_hid)))
        self.W_out = np.random.uniform(low=np.zeros((self.N_out, self.N_hid)), 
                                   high=np.ones((self.N_out, self.N_hid)))
        self.r_history = np.zeros((self.N_hid))
        self.mem = np.zeros((self.N_hid))
        # self.spike = np.zeros((self.N_hid))
    
    def membrane(self, x, spike):
        mem = self.mem * self.decay * (1-spike) + x
        spike = np.array(mem>self.thr, dtype=np.float)
        self.mem = mem
        return spike

    def activation(self, x):
        return np.tanh(x)
    
    def forward(self, x):
        assert x.shape[0]>1
        spike_train = []
        spike = np.zeros((self.N_hid))
        timestep = x.shape[0]
        for t in range(timestep):
            Ar = np.matmul(self.A, self.r_history)
            U = np.matmul(self.W_in, x[t,:])
            r = (1-self.alpha) * self.r_history + self.alpha * self.activation(Ar + U)
            spike = self.membrane(r, spike)
            spike_train.append(spike)
            self.r_history = r
            
        y = np.matmul(self.W_out, r)
        return y, spike_train

def data_generation():
    train_data = torchvision.datasets.MNIST(root='./reservoir/data/',
                                            train=True,
                                            transform=None,
                                            download=False,
                                            ).data.float()/255
    data = []
    for _ in range(30):
        img = Uniform(low=torch.zeros(28,28), high=torch.ones(28,28)).sample()
        idx = img<=(train_data[1000]-0.5)
        img = torch.zeros_like(img)
        img[idx] = 1
        data.append(img.flatten().numpy())
    return np.array(data)


if __name__ == '__main__':
    data = data_generation()
    model = RC(N_input=28*28,
               N_hidden=1000,
               N_output=10,
               alpha=0.2,
               decay=0.5,
               threshold=0.7,
               )
    print(data.shape)
    y, spike_train = model.forward(data)
    spike_train = np.array(spike_train)
    print(y, spike_train.shape)
    plt.imshow(spike_train[:,0:100])
    plt.pause(10)
