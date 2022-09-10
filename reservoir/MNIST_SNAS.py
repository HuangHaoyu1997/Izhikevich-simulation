# -*- coding: utf-8 -*-
"""
    The Neural Structure Search (NAS) of large scale Liquid State Machine
    (LSM) for MNIST. The optimization method adopted
    here is CMA-ES, BO and Gaussian process assisted CMA-ES.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.

Requirement
=======
Numpy
Pandas
Brian2

Usage
=======

Citation
=======

"""
from src import *

from functools import partial
from multiprocessing import Pool

from brian2 import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
prefs.codegen.target = "numpy"
start_scope()
np.random.seed(100)
data_path = './data/MNIST/raw/' # '../../../Data/MNIST_data/'
###################################
# -----simulation parameter setting-------
coding_n = 3           # cos编码基函数数量
MNIST_shape = (1, 784)
coding_duration = 30   # 编码序列的长度
duration = coding_duration * MNIST_shape[0] # 30

F_train = 0.05 # 60000*0.05=3000
F_validation = 0.00833333 # 60000 * 0.00833333 = 500
F_test = 0.05
Dt = defaultclock.dt = 1 * ms

# -------class initialization----------------------
function = MathFunctions()
base = BaseFunctions()
readout = Readout()
MNIST = MNIST_classification(MNIST_shape, duration)

# -------data initialization----------------------
MNIST.load_Data_MNIST_all(data_path)
df_train_validation = MNIST.select_data(F_train + F_validation, MNIST.train)
df_train, df_validation = train_test_split(df_train_validation, test_size=F_validation / (F_validation + F_train), random_state=42)
df_test = MNIST.select_data(F_test, MNIST.test)

df_en_train = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_train, coding_n)
df_en_validation = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_validation, coding_n)
df_en_test = MNIST.encoding_latency_MNIST(MNIST._encoding_cos_rank_ignore_0, df_test, coding_n)

data_train_s, label_train = MNIST.get_series_data_list(df_en_train, is_group=True)
data_validation_s, label_validation = MNIST.get_series_data_list(df_en_validation, is_group=True)
data_test_s, label_test = MNIST.get_series_data_list(df_en_test, is_group=True)

# -------get numpy random state------------
np_state = np.random.get_state()

############################################
# ---- define network run function----
def run_net(inputs, **parameter):
    """
    inputs: list, len=2
    inputs 是1个样本
    inputs[0].shape = (30, 2352)
    inputs[1] label
    
    
    Parameters = [R, p_inE/I, f_in, f_EE, f_EI, f_IE, f_II, tau_ex, tau_inh]
    ----------
    f_in: 线性输入层——池化层的突触连接强度参数
    f_EE: 兴奋性——兴奋性,突触强度参数
    f_EI: 兴奋性——抑制性,突触强度参数
    f_IE:
    f_II:
    tau_ex: 兴奋性神经元的膜时间常数
    tau_inh: 抑制性神经元的膜时间常数
    
    """

    # ---- set numpy random state for each run----
    np.random.set_state(np_state)
    print(len(inputs), inputs[1])

    # -----parameter setting-------
    n_ex = 1600             # 兴奋性神经元数量
    n_inh = int(n_ex / 4)   # 抑制性神经元数量
    n_input = MNIST_shape[1] * coding_n # 784*3=2352
    n_read = n_ex + n_inh   # 1600+400=2000

    R = parameter['R']
    f_in = parameter['f_in']
    f_EE = parameter['f_EE']
    f_EI = parameter['f_EI']
    f_IE = parameter['f_IE']
    f_II = parameter['f_II']

    A_EE = 60 * f_EE
    A_EI = 60 * f_EI
    A_IE = 60 * f_IE
    A_II = 60 * f_II
    A_inE = 60 * f_in
    A_inI = 60 * f_in

    tau_ex = parameter['tau_ex'] * coding_duration
    tau_inh = parameter['tau_inh'] * coding_duration
    tau_read = 30

    p_inE = parameter['p_in'] * 0.1 # 线性输入层神经元与兴奋性神经元的连接概率
    p_inI = parameter['p_in'] * 0.1 # 线性输入层神经元与抑制性神经元的连接概率

    # ------definition of equation-------------
    neuron_in = '''I = stimulus(t,i) : 1'''

    neuron = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1 (unless refractory)
    dg/dt = (-g)/(3*ms) : 1
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h)+13.5: 1
    x : 1
    y : 1
    z : 1
    '''

    neuron_read = '''
    tau : 1
    dv/dt = (I-v) / (tau*ms) : 1
    dg/dt = (-g)/(3*ms) : 1 
    dh/dt = (-h)/(6*ms) : 1
    I = (g+h): 1
    '''

    synapse = '''w : 1'''

    on_pre_ex = '''g+=w'''  # 兴奋性神经元，作为突触前神经元发放，将执行的代码
    
    on_pre_inh = '''h-=w''' # 抑制性神经元，作为突触前神经元发放，将执行的代码

    # -----Neurons setting-------
    Input = NeuronGroup(n_input, 
                        neuron_in, 
                        threshold='I > 0', 
                        method='euler', 
                        refractory=0 * ms,
                        name='neurongroup_input')
    
    # excitatory兴奋性神经元
    G_ex = NeuronGroup(n_ex, 
                        neuron, 
                        threshold='v > 15', 
                        reset='v = 13.5', 
                        method='euler', 
                        refractory=3 * ms,
                        name='neurongroup_ex')

    # inhibitory异质性神经元
    G_inh = NeuronGroup(n_inh, 
                        neuron, 
                        threshold='v > 15', 
                        reset='v = 13.5', 
                        method='euler', 
                        refractory=2 * ms,
                        name='neurongroup_in')
    # 读出层
    G_readout = NeuronGroup(n_read, neuron_read, method='euler', name='neurongroup_read')

    # -----Synapses setting-------
    # 输入层——兴奋性
    S_inE = Synapses(Input, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inE')
    # 输入层——抑制性
    S_inI = Synapses(Input, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_inI')
    # 兴奋性——兴奋性
    S_EE = Synapses(G_ex, G_ex, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EE')
    # 兴奋性——抑制性
    S_EI = Synapses(G_ex, G_inh, synapse, on_pre=on_pre_ex, method='euler', name='synapses_EI')
    # 抑制性——兴奋性
    S_IE = Synapses(G_inh, G_ex, synapse, on_pre=on_pre_inh, method='euler', name='synapses_IE')
    # 抑制性——抑制性
    S_II = Synapses(G_inh, G_inh, synapse, on_pre=on_pre_inh, method='euler', name='synapses_I')
    # 兴奋性——读出层
    S_E_readout = Synapses(G_ex, G_readout, 'w = 1 : 1', on_pre=on_pre_ex, method='euler', name='synapses_Er')
    # 抑制性——读出层
    S_I_readout = Synapses(G_inh, G_readout, 'w = 1 : 1', on_pre=on_pre_inh, method='euler', name='synapses_Ir')

    # -------initialization of neuron parameters----------
    G_ex.v = '13.5+1.5*rand()'
    G_inh.v = '13.5+1.5*rand()'
    G_readout.v = '0'
    
    G_ex.g = '0'
    G_inh.g = '0'
    G_readout.g = '0'
    
    G_ex.h = '0'
    G_inh.h = '0'
    G_readout.h = '0'
    
    G_ex.tau = tau_ex # 时间常数
    G_inh.tau = tau_inh
    G_readout.tau = tau_read
    # 给神经元分配XYZ坐标
    [G_ex, G_inh] = base.allocate([G_ex, G_inh], 10, 10, 20)

    # -------initialization of network topology and synapses parameters----------
    S_inE.connect(condition='j<0.3*N_post', p=p_inE)
    S_inI.connect(condition='j<0.3*N_post', p=p_inI)
    # 将neuron分配至3维空间，计算欧氏距离决定连接概率
    S_EE.connect(condition='i != j', p='0.3*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_EI.connect(p='0.2*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_IE.connect(p='0.4*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_II.connect(condition='i != j', p='0.1*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/R**2)')
    S_E_readout.connect(j='i')
    S_I_readout.connect(j='i+n_ex')

    S_inE.w = function.gamma(A_inE, S_inE.w.shape) # gamma分布随机变量采样
    S_inI.w = function.gamma(A_inI, S_inI.w.shape)
    S_EE.w = function.gamma(A_EE, S_EE.w.shape)
    S_IE.w = function.gamma(A_IE, S_IE.w.shape)
    S_EI.w = function.gamma(A_EI, S_EI.w.shape)
    S_II.w = function.gamma(A_II, S_II.w.shape)

    S_EE.pre.delay = '1.5*ms'
    S_EI.pre.delay = '0.8*ms'
    S_IE.pre.delay = '0.8*ms'
    S_II.pre.delay = '0.8*ms'

    # ------create network-------------
    net = Network(collect())
    net.store('init')

    # ------run network-------------
    stimulus = TimedArray(inputs[0], dt=Dt)
    net.run(duration * Dt)
    states = net.get_states()['neurongroup_read']['v']
    net.restore('init')
    return (states, inputs[1])


@Timelog
@AddParaName
def parameters_search(**parameter):
    '''
    return: error rate越低越好
    '''
    # ------parallel run for train-------
    states_train_list = pool.map(partial(run_net, **parameter), 
                                 [(x) for x in zip(data_train_s, label_train)])
    
    # ------parallel run for validation-------
    states_validation_list = pool.map(partial(run_net, **parameter),
                                      [(x) for x in zip(data_validation_s, label_validation)])
    
    # ----parallel run for test--------
    states_test_list = pool.map(partial(run_net, **parameter), 
                                [(x) for x in zip(data_test_s, label_test)])
    
    # ------Readout---------------
    states_train, states_validation, states_test, _label_train, _label_validation, _label_test = [], [], [], [], [], []
    for train in states_train_list:
        states_train.append(train[0]) # inference
        _label_train.append(train[1]) # ground truth label
    for validation in states_validation_list:
        states_validation.append(validation[0])
        _label_validation.append(validation[1])
    for test in states_test_list:
        states_test.append(test[0])
        _label_test.append(test[1])
        
    states_train = (MinMaxScaler().fit_transform(np.asarray(states_train))).T
    states_validation = (MinMaxScaler().fit_transform(np.asarray(states_validation))).T
    states_test = (MinMaxScaler().fit_transform(np.asarray(states_test))).T
    score_train, score_validation, score_test = readout.readout_sk(X_train=states_train, 
                                                                   X_validation=states_validation, 
                                                                   X_test=states_test,
                                                                   y_train=np.asarray(_label_train),
                                                                   y_validation=np.asarray(_label_validation),
                                                                   y_test=np.asarray(_label_test), 
                                                                   # for logistic regression
                                                                   solver="lbfgs",
                                                                   multi_class="multinomial",
                                                                   )
    # ----------show results-----------
    print('parameters %s' % parameter)
    print('Train score: ', score_train)
    print('Validation score: ', score_validation)
    print('Test score: ', score_test)
    return 1 - score_validation, 1 - score_test, 1 - score_train, parameter


##########################################
# -------optimizer settings---------------

if __name__ == '__main__':
    core = 8
    pool = Pool(core)
    parameters = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    bounds = {'R': (0.0001, 1), 
              'p_in': (0.0001, 1), 
              'f_in': (0.0001, 1), 
              'f_EE': (0.0001, 1), 
              'f_EI': (0.0001, 1),
              'f_IE': (0.0001, 1), 
              'f_II': (0.0001, 1), 
              'tau_ex': (0.0001, 1), 
              'tau_inh': (0.0001, 1),
              }
    parameters_search.func.keys = list(bounds.keys())

    LHS_path = './Results_Record.dat' # './LHS_MNIST.dat'
    SNAS = 'SAES'

    # -------parameters search---------------
    if SNAS == 'BO':
        optimizer = BayesianOptimization_(
            f=parameters_search,
            pbounds=bounds,
            random_state=np.random.RandomState(),
        )

        logger = bayes_opt.observer.JSONLogger(path="./BO_res_MNIST.json")
        optimizer.subscribe(bayes_opt.event.Events.OPTMIZATION_STEP, logger)

        optimizer.minimize(
            LHS_path=LHS_path,
            init_points=50,
            is_LHS=True,
            n_iter=250,
            acq='ei',
            opt=optimizer.acq_min_DE,
        )

    elif SNAS == 'SAES':
        saes = SAES(f=parameters_search, 
                    acquisition='ei', 
                    x0=parameters, 
                    sigma0=0.5,
                    **{'ftarget': -1e+3, 
                       'bounds': bounds, 
                       'maxiter': 500,
                       'tolstagnation': 500
                       })
        saes.run_best_strategy(50, 1, 2, LHS_path=None)

    elif SNAS == 'CMA':
        res = cma.fmin(parameters_search, parameters, 0.5,
                       options={'ftarget': -1e+3, 'maxiter': 30,
                                'bounds': np.array([list(x) for x in list(bounds.values())]).T.tolist()})