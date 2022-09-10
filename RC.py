import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os, time, torch, ray, pickle
from scipy.linalg import pinv
import torchvision.transforms as transforms
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RC:
    def __init__(self,
                 N_input, # 输入维度
                 N_hidden, # reservoir神经元数量
                 N_output, # 输出维度
                 alpha, # 
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
        # 用系数0.0533缩放，以保证谱半径ρ(A)=1.0
        self.W_out = np.random.uniform(low=-0.0533*np.ones((self.N_out, self.N_hid)), 
                                       high=0.0533*np.ones((self.N_out, self.N_hid)))
        self.r_history = np.zeros((self.N_hid))
        self.mem = np.zeros((self.N_hid))
        # self.spike = np.zeros((self.N_hid))
    
    def state_dict(self,):
        return {
            'W_in': self.W_in,
            'A': self.A,
            'W_out': self.W_out,
            'N_input': self.N_in,
            'N_hidden': self.N_hid,
            'N_output': self.N_out,
            'alpha': self.alpha,
            'decay': self.decay,
            'threshold': self.thr,
        }
        
    def membrane(self, x, spike):
        mem = self.mem * self.decay * (1-spike) + x
        spike = np.array(mem>self.thr, dtype=np.float32)
        self.mem = mem
        return spike

    def activation(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        return np.exp(x)/np.exp(x).sum()
    
    def forward(self, x):
        '''
        一个样本的长度应该超过1,即由多帧动态数据构成
        '''
        assert x.shape[0]>1
        spike_train = []
        spike = np.zeros((self.N_hid))
        timestep = x.shape[0]
        for t in range(timestep):
            Ar = np.matmul(self.A, self.r_history)
            U = np.matmul(self.W_in, x[t,:])
            r = (1 - self.alpha) * self.r_history + self.alpha * self.activation(Ar + U)
            spike = self.membrane(r, spike)
            spike_train.append(spike)
            self.r_history = r
            
        y = np.matmul(self.W_out, r)
        y = self.softmax(y)
        return r, y, spike_train

def cross_entropy(p, q):
    '''
    交叉熵
    CELoss = -∑ p(x)*log(q(x))
    '''
    return -(p * np.log(q)).sum()

@ray.remote
def inference(model:RC,
              train_loader,
              frames):
    rs = []
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = encoding(images.squeeze(), frames) # shape=(30,784)
        r, y, spike = model.forward(images)
        rs.append(r)
        
    print('Time elasped:', time.time()-start_time)
    return np.array(rs)

@ray.remote
def train(model:RC, 
          solution,
          train_loader, 
          test_loader, 
          batch_size, 
          frames,
          ):

    model.W_out = solution.reshape(model.N_out, model.N_hid)
    running_loss = 0
    start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        images = encoding(images.squeeze(), frames) # shape=(30,784)
        r, outputs, _ = model.forward(images)
        
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1).squeeze().numpy()
        loss = cross_entropy(labels_, outputs)
        running_loss += loss
        
    print('Time elasped:', time.time()-start_time)
    return running_loss / len(train_loader)
        
'''
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs = encoding(inputs.squeeze(), frames=frames) # shape=(30,784)
    outputs, _ = model.forward(inputs)
    labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1).squeeze().numpy()
    loss = cross_entropy(labels_, outputs)
    predicted = outputs.argmax()
    total += float(targets.size(0))
    correct += float(predicted==targets.item())
    if batch_idx %100 ==0:
        acc = 100. * float(correct) / float(total)
        print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

print('Iters:', epoch,'\n\n\n')
print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
acc = 100. * float(correct) / float(total)
acc_record.append(acc)
if epoch % 5 == 0:
    print(acc)
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_record': acc_record,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt' + names + '.t7')
    best_acc = acc
'''
def spectral_radius(M):
    a,b = np.linalg.eig(M) #a为特征值集合，b为特征值向量
    return np.max(np.abs(a)) #返回谱半径

def encoding(image, frames):
    '''
    随机分布编码
    frames:动态帧长度
    '''
    sample = []
    for _ in range(frames):
        img = (image > torch.rand(image.size())).float().flatten().numpy()
        sample.append(img)
    return np.array(sample)

def MNIST_generation(batch_size):
    '''
    生成随机编码的MNIST动态数据集
    '''
    
    train_dataset = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                               train=True, 
                                               download=False, 
                                               transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=0)

    test_set = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                          train=False, 
                                          download=False, 
                                          transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              num_workers=0)
    return train_loader, test_loader



if __name__ == '__main__':
    ray.init()
    
    train_loader, test_loader = MNIST_generation(batch_size=1)
    model = RC(N_input=28*28,
               N_hidden=1000,
               N_output=10,
               alpha=0.2,
               decay=0.5,
               threshold=0.7,
               )
    from cma import CMAEvolutionStrategy
    es = CMAEvolutionStrategy(x0=np.zeros((model.N_hid*model.N_out)),
                                sigma0=0.5,
                                #   inopts={
                                #             'popsize':100,
                                #           },
                                )
    N_gen = 100
    for g in range(N_gen):
        solutions = es.ask()
        task_list = [train.remote(model,
                                    solution,
                                    train_loader, 
                                    test_loader, 
                                    batch_size=1,
                                    frames=10,
                                    ) for solution in solutions]
        fitness = ray.get(task_list)
        es.tell(solutions, fitness)
        with open('ckpt_'+str(g)+'.pkl', 'wb') as f:
            pickle.dump([solutions, fitness], f)
        print(np.min(fitness))
    
    # labels = []
    # for i, (image, label) in enumerate(train_loader):
    #     label_ = torch.zeros(1, 10).scatter_(1, label.view(-1, 1), 1).squeeze().numpy()
    #     labels.append(label_)
    # labels = np.array(labels, dtype=np.float).T
    
    # with open('train_labels.pkl', 'rb') as f:
    #     labels = pickle.load( f)
        
    # with open('rs.pkl', 'rb') as f:
    #     R_T = pickle.load(f)
    
    # R = R_T.T
    # R_inv = pinv(R)
    # W_out = np.matmul(labels, R_inv)
    # print(W_out.shape)
    # model.W_out = W_out
    
    # correct = 0
    # for i, (image, label) in enumerate(train_loader):
    #     image = encoding(image.squeeze(), 10) # shape=(30,784)
    #     r, y, _ = model.forward(image)
    #     predict = y.argmin()
    #     correct += float(predict == label.item())
    # print(correct / len(train_loader))
    
    # rs = inference.remote(model, train_loader, 10)
    # rs = ray.get(rs)
    # 
    # print(rs.shape)
    # with open('rs.pkl', 'wb') as f:
    #     pickle.dump(rs, f)
    
    # y, spike_train = model.forward(data)
    # spike_train = np.array(spike_train)
    # print(y, spike_train.shape)
    # plt.imshow(spike_train[:,0:100])
    # plt.pause(10)
    ray.shutdown() 
