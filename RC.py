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
                 ) -> None:
        self.N_in = N_input
        self.N_hid = N_hidden
        self.N_out = N_output
        self.alpha = alpha
        self.reset()
        
    def reset(self,):
        self.W_in = np.random.uniform(low=np.zeros((self.N_hid, self.N_in)), 
                                      high=np.ones((self.N_hid, self.N_in))*0.1)
        self.A = np.random.uniform(low=-1*np.ones((self.N_hid, self.N_hid)), 
                                   high=np.ones((self.N_hid, self.N_hid)))
        self.W_out = np.random.uniform(low=np.zeros((self.N_out, self.N_hid)), 
                                   high=np.ones((self.N_out, self.N_hid)))
        self.r_history = np.zeros((self.N_hid))
        
        
    def activation(self, x):
        return np.tanh(x)
    
    def forward(self, x):
        assert x.shape[0]>1
        timestep = x.shape[0]
        for t in range(timestep):
            Ar = np.matmul(self.A, self.r_history)
            U = np.matmul(self.W_in, x[t,:])
            r = (1-self.alpha) * self.r_history + self.alpha * self.activation(Ar + U)
            self.r_history = r


        y = np.matmul(self.W_out, r)
        return y

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
               )
    print(data.shape)
    y = model.forward(data)
    print(y)
