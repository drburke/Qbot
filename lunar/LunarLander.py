import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import math
import time
import threading

random.seed()
# %matplotlib inline

import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#  Based in part on https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

ENV = 'LunarLander-v2'

RUN_TIME = 30
NUM_THREADS = 8
NUM_OPTIMIZERS = 2
THREAD_DELAY = 0.001


GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = 0.5
LOSS_ENTROPY = 0.01

ALPHA = 0.1

def leaky_relu(x,alpha=0.02):
    return tf.maximum(alpha*x,x)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


# Create class QNetwork
class A3CNetwork:

    train_queue = [ [], [], [], [], [] ] # s, a r, s', s' mask
    lock_queue = threading.Lock()

    def __init__(self, \
                 learning_rate=0.01, \
                 state_size=STATE_SIZE, 
                 action_size=ACTION_SIZE, \
                 hidden_size=10, \
                 hidden_layers=2, \
                 alpha=ALPHA):
        
        # Dropout
        self.keep_prob_ = tf.placeholder(tf.float32,name='keep_prob')
    
        # State
        self.state_ = tf.placeholder(tf.float32,[None, state_size],name='state')
        
        # Actions, not one hot
        self.actions_ = tf.placeholder(tf.int32,[None],name='actions')

        # Actions, one hot
        self.one_hot_actions = tf.one_hot(self.actions_, action_size)
        
        # R value
        self.R_ = tf.placeholder(tf.float32,[None,1],name='R')
        
        self.value_ = tf.placeholder(tf.float32,[None,1],name='value_input')
        
        with tf.variable_scope("encoder"):
#             self.fcl_weights = tf.Variable(tf.truncated_normal((state_size, hidden_size), mean=0.0, stddev=0.1),name='weights') 
#             self.fcl_bias = tf.Variable(tf.zeros(hidden_size),name="bias")
#             self.fcl_sum = tf.add(tf.matmul(self.state_, self.fcl_weights), self.fcl_bias)
#             self.fcl_relu = leaky_relu(fcl)
            self.fcl = tf.layers.dense(self.state_, hidden_size,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.fcl = leaky_relu(self.fcl)

    
        with tf.variable_scope("policy"):
#             self.policy_weights = tf.Variable(tf.truncated_normal((hidden_size,action_size)),name="weights")
#             self.policy_bias = tf.Variable(tf.zeros(action_size),name="bias")
#             self.policy = tf.add(tf.matmul(self.fcl_relu,self.policy_weights),self.policy_bias)
            self.policy = tf.layers.dense(self.fcl, action_size,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='policy_out')
        
        self.policy_softmax = tf.nn.softmax(self.policy,name='policy_softmax_out')
        self.log_policy_softmax = tf.log(tf.reduce_sum(self.policy_softmax*self.actions_) + 1e-10,name='policy_log_softmax_out')
        
        with tf.variable_scope("value"):
#             self.value_weights = tf.Variable(tf.truncated_normal((hidden_size,1)),name="weights")
#             self.value_bias = tf.Variable(tf.zeros(1),name="bias")
#             self.value = tf.add(tf.matmul(self.fcl_relu,self.value_weights),self.value_bias)
            self.value_layer = tf.layers.dense(self.fcl, 1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.value = tf.identity(self.value_layer,name='value')

        self.policy_loss = -tf.reduce_mean(tf.multiply(self.log_policy_softmax , (self.R_ - self.value_)))

        self.value_loss = tf.reduce_mean(tf.square(self.R_ - self.value))
        

        self.entropy = LOSS_ENTROPY * tf.reduce_sum(self.policy_softmax*tf.log(self.policy_softmax + 1e-10)))

        self.total_loss = tf.reduce_mean(self.policy_loss + self.value_loss + self.entropy)

       
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.99).minimize(self.total_loss)

    def optimize(self):

        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:
                return

            s, a, r, s_, s_mask = self.train_queue

            self.train_queue = [ [], [], [], [], [] ]


        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH: print('Alert ! Minimizing large batch of size %d' % len(s))

        v = self.predict_v(s_)

        r  = r + GAMMA_N * v * s_mask

        self.sess.run(self.optimizer, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:

            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None: 
                self.train_queue[3].append(NONE_STATE)  # Next state
                self.train_queue[4].append(0.) # Mask
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):

        policy, value = self.session.run([self.policy_softmax,self.value], feed_dict{self.state: s})

        return policy, value

    def predict_p(self, s):
        policy = self.session.run(self.policy_softmax, feed_dict{self.state: s})

        return policy

    def predict_p(self, s):
        value = self.session.run(self.value, feed_dict{self.state: s})

        return value
# create memory class for storing previous experiences
class Memory():
    def __init__(self, max_size = 10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=True)
        return [self.buffer[ii] for ii in idx]
    
    def pull_all(self):
        return self.buffer
    
    def clear(self):
        self.buffer = deque(maxlen=self.max_size)

def normalize_state(x, denormalize=False):
    # Rough max/min extent for states, normalize to +/- 1
    # [-1,1] [-0.2,1.2] [-2,2] [0.5,-2]  [3.5,-3.5]  [6,-6] [1,0]  [1,0]
    y = x / [1.,1.,2.,1.5,3.5,6.,1.,1.]
    return y



class Agent:
    def __init__(self):
        self.memory = []
        self.R = 0.

    def act(self, s):

        s = np.array([s])
        p = brain.predict_p(s)[0]

        a = np.random.choice(ACTION_SIZE, p=p)

        return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        
        self.memory.append((s, a, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, ar, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)




class Environment(threading.Thread):

    stop_signal = False

    def __init__(self, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent()

    def runEpisode(self):
        s = self.env.reset()
        R = 0
        while True:
            time.sleep(THREAD_DELAY)

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


env_test = Environment(render=True)
STATE_SIZE = env_test.env.observation_space.shape[0]
ACTION_SIZE = env_test.env.action_space.n
NONE_STATE = np.zeros(STATE_SIZE)


brain = A3CNetwork()

envs = [Environment() for i in range(NUM_THREADS)]
opts = [Optimizer() for i in range(NUM_OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Done training")
env_test.run()



def train_a3c_network(train_episodes=500,\
                   gamma=0.99,\
                   explore_start=1.0,\
                   explore_stop=0.01,\
                   decay_rate=0.0001,\
                   hidden_size=64,\
                   hidden_layers=2,\
                   learning_rate=0.0001,\
                   memory_size=10000,\
                   batch_size=20,\
                   max_steps=5000,\
                   alpha=0.1,\
                   verbose=True,\
                   num_trains=50,\
                   num_bots=16,\
                   action_size=4):
    
    #loaded_graph = tf.Graph()
    # Create the network
    mainQN = A3CNetwork(name='main', hidden_size=hidden_size, hidden_layers=hidden_layers, learning_rate=learning_rate, alpha=alpha)
    
    # Memory for asynchronous replay
    memory = Memory(max_size=memory_size)
    
    # Reset state, normalize
    state = env.reset()
    state = normalize_state(state)
    
    #  Create output variables
    total_rewards_tf = tf.placeholder(tf.float32, None, name='total_rewards')
    max_q_tf = tf.placeholder(tf.float32, None, name='max_qs')

    # Add scalar summary trackers
    tf.summary.scalar('total_reward', total_rewards_tf)
    tf.summary.scalar('max_q', max_q_tf)
    merged_tf = tf.summary.merge_all()
    

    
    saver = tf.train.Saver()
    rewards_step_list = []
    

    mainQN.sess = tf.InteractiveSession()
    assert mainQN.sess.graph is tf.get_default_graph()
        
    # Initialize variables
    mainQN.sess.run(tf.global_variables_initializer())

    save_file_name = 'checkpts'
    saver.save(mainQN.sess,'checkpoints/'+save_file_name)
    mainQN.sess.close()

    mainQN.sess = tf.InteractiveSession()
    assert mainQN.sess.graph is tf.get_default_graph()
    saver.restore(mainQN.sess,'checkpoints/'+save_file_name)


    # Create file writer
    file_writer = tf.summary.FileWriter(log_path,mainQN.sess.graph)
    
    step = 0
    rewards_list = []
    
    for ep in range(train_episodes):
        
        
        do_render = os.path.isfile('./render.txt')
        biggest_target = -9e9
        smallest_target = 9e9
        done = 0
        memory.clear()
        
        for bot in range(num_bots):
            total_reward = 0
            t = 0
            prev_reward = 0
            R = 0
            while not done:
                step += 1
                
                if do_render:
                    env.render()  

                
                # Get action from policy-network
                feed = {mainQN.state_: state.reshape((1, *state.shape))}
                Qs,Qraw,value = mainQN.sess.run([mainQN.policy_softmax,mainQN.policy,mainQN.value], feed_dict=feed)
                action = np.argmax(Qs)
                
                # Choose random action based on softmax probabilities
                rand = np.random.rand()
                action = 0
                sum_iter = Qs[0,action]
#                     print(Qs)
#                     print(num_iter)
#                     print(sum_iter)
                while sum_iter < rand:
                    action += 1
#                         print(num_iter)
#                         print(Qs[num_iter])
#                         print('sum=',sum_iter,'  rand=',rand)
                    sum_iter += Qs[0,action]
#                         if sum_iter >= rand:
#                             print('final sum=',sum_iter,'  rand=',rand)
#                             break                    

                if bot is 0 and t is 0:
                    print('Qs=',Qs,'  value=',value,'  a=',action)
                Qraw_max = np.max(Qraw)
                Qraw_min = np.min(Qraw)
                biggest_target = np.maximum(Qraw_max,biggest_target)
                smallest_target = np.minimum(Qraw_min,smallest_target)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)
                # if np.abs(reward)<0.3:
                #     reward -= -0.1
                if math.isnan(reward):
                    print('reward was nan !')
                new_reward = reward
#                     reward -= prev_reward
                prev_reward = new_reward
                R = reward + gamma * R
                
                next_state = normalize_state(next_state)
                state = next_state
                total_reward += reward
                t += 1
                               

                # Add experience to memory
                memory.add((state, Qs, reward, done, value, R))

            
            rewards_step_list.append(total_reward)
            state = env.reset()
            state = normalize_state(state)  
            done = 0
                
        

          # Sample mini-batch from memory
        batch = memory.pull_all()
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        dones = np.array([each[3] for each in batch])
        values = np.array([each[4] for each in batch])
        Rs = np.array([each[5] for each in batch])
        
#             # Now bot updates gradients
#             for i in range(len(dones)):

#                 if dones[i] == 1:
#                     R = 0
#                 else:
#                     R = rewards[i] + gamma * R

#                 print('R=',R,'  reward[i]=',rewards[i])
#                 memory.add((state, Qs, reward, done, value))
            
#             print('states=',np.shape(states))
#             print('actions=',np.shape(actions))
        re_actions = np.squeeze(actions)

        re_values = np.squeeze(values,axis=2)

        re_rs = np.reshape(Rs,(len(Rs),1))
        
        policy_loss, _, value_loss, _ = mainQN.sess.run([mainQN.policy_loss,mainQN.policy_opt,mainQN.value_loss,mainQN.value_opt],
                                                feed_dict={mainQN.state_:states,
                                                          mainQN.R_:re_rs,
                                                          mainQN.value_:re_values})
# ,
#                                                               mainQN.policy_softmax:re_actions,
#                                                               mainQN.value:re_values

        total_reward = total_reward
        rewards_list.append((ep, total_reward))   
        runningMean = np.mean(rewards_step_list[-100:])

        summary = mainQN.sess.run(merged_tf, feed_dict={total_rewards_tf: total_reward, 
                                                 max_q_tf: biggest_target})
        file_writer.add_summary(summary,ep)
        if verbose:

            print('Episode: {}'.format(ep),
                  'TReward: {}'.format(total_reward),
                  'RunMean : {:.4f}'.format(runningMean),
                  'MaxTarg : {:.4f}'.format(biggest_target),
                  'MinTarg : {:.4f}'.format(smallest_target))
               
#             if ep>0:
#                 return rewards_list, mainQN, saver, runningMean
    saver.save(mainQN.sess, "checkpoints/cartpole.ckpt")
    mainQN.sess.close()
    return rewards_list, mainQN, saver, runningMean


def plot_rewards(rewards_list):
    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')


def test_and_train_qnetwork(train_episodes=1000,\
                   gamma=0.99,\
                   explore_start=1.0,\
                   explore_stop=0.01,\
                   decay_rate=0.0001,\
                   hidden_size=64,\
                   hidden_layers=2,\
                   learning_rate=0.0001,\
                   memory_size=10000,\
                   batch_size=20,\
                   test_episodes=10,\
                   render=False,\
                   alpha=0.,\
                   verbose=True,\
                   num_trains=50):
    
    # reset graph
    tf.reset_default_graph()

    # train q-network
    rewards_list, mainQN, saver, runningMean = train_a3c_network(train_episodes = train_episodes, \
                                                  gamma=gamma,\
                                                  explore_start=explore_start,\
                                                  explore_stop=explore_stop,\
                                                  decay_rate=decay_rate,\
                                                  hidden_size=hidden_size,\
                                                  hidden_layers=hidden_layers,\
                                                  learning_rate=learning_rate,\
                                                  memory_size=memory_size,\
                                                  batch_size=batch_size,\
                                                  alpha=alpha,\
                                                  verbose=verbose,\
                                                  num_trains=num_trains)

    if verbose:
        # plot training
        plot_rewards(rewards_list)
    
    avg_train_rewards = np.sum([each[1] for each in rewards_list]) / len(rewards_list)
    
    if verbose:
        print('average training reward = ',avg_train_rewards)

    
    return avg_train_rewards, mainQN, saver, len(rewards_list), runningMean


from gym import wrappers
log_path = './logs/3/logs_run_drop=1.0_-=0.1'  

env = gym.make('LunarLander-v2')
#env = wrappers.Monitor(env, '/tmp/lunarlander-experiment-2',force=True)

train, mainQN, saver, num_episodes, runningMean = test_and_train_qnetwork(memory_size=100000,\
                                     train_episodes=10000,\
                                           gamma=0.99,\
                                           explore_start=1.,\
                                           explore_stop=0.1,\
                                           decay_rate=0.0001,\
                                           hidden_layers=1,\
                                           hidden_size=256,\
                                           learning_rate=0.0001,\
                                           batch_size=128,\
                                           alpha=0.1,\
                                           num_trains = 128,\
                                           verbose=True)
print('train=',str(train))
print('number of episodes=',str(num_episodes))
env.close()