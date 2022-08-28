import numpy as np
import matplotlib.pyplot as plt

def encode(x, train_len, base, min, max):
    '''
    x: input data
    train_len: spike train length
    base: number of cosine functions
    '''
    def projection(x, min, max):
        '''
        project x into [0, pi]
        '''
        return np.pi * (x - min) / (max - min)
    
    x_shape = x.shape
    x_ = projection(x, min, max)
    spike_train = np.zeros((*x_shape, train_len), dtype=np.float32)
    y = np.array([train_len * np.cos(x_ + i*np.pi/base)**2 for i in range(base)])
    print(y, y.shape)
    # np.cos()
    return y

if __name__ == '__main__':
    input = np.random.rand(5,5)
    # print(np.cos(input))
    # print(np.cos(input+2*np.pi))
    inpt = np.arange(0, 10, 0.1)
    enc = encode(inpt, 10, 5, 0, 1)
    # plt.plot(inpt, enc[0,:])
    # plt.plot(inpt, enc[1,:])
    # plt.plot(inpt, enc[2,:])
    # plt.plot(inpt, enc[3,:])
    # plt.plot(inpt, enc[4,:])
    # plt.show()
    print(np.round(enc[:,2]))
