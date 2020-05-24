#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:10:37 2020

@author: felipeolivosvaldivia
"""

import planilla_env
from time import time

t0 = time()
#ruta mac/parallels
#ruta = '//mac/Google Drive/Colab Notebooks/Diseno Fundacion/Fund Interruptor (python).xlsx'
#ruta pc escritorio
ruta = 'Fund Interruptor (python).xlsx'
#ruta Spyder mac
#ruta = '/Users/felipeolivosvaldivia/Google Drive/Colab Notebooks/Diseno Fundacion/Fund Interruptor (python).xlsx'
env = planilla_env.design_env(ruta)

#_______________________________________
#NN implementation

import copy
import torch
import numpy as np
import random
from matplotlib import pylab as plt


def one_hot(N,pos, val=1):
  one_hot_vec = np.zeros(N)
  one_hot_vec[pos] = val
  return one_hot_vec


l1 = len(env.get_state())
l2 = 150
l3 = 100
l4 = 12 #number of possible actions


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model) #A
model2.load_state_dict(model.state_dict()) #B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-2 #1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.8#0.9
epsilon = 0.3

#_______________________________________
#NN implementation

from collections import deque
epochs = 300
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
max_moves = 50#50
h = 0
sync_freq = 500 #
j=0
for i in range(epochs):
    game = env
    game.set_start()
    state1_ = np.resize(np.array(env.get_state()), (1,45))                      #Acá hicimos un cambio
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while(status == 1):
        j+=1
        mov += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = random.randint(0,11)
        else:
            action_ = np.argmax(qval_)
        
        action_hot = one_hot(l4, action_)
        game.apply_action(action_hot)
        state2_ = np.resize(np.array(game.get_state()),(1,45))                  #Acá hicimos un cambio
        state2 = torch.from_numpy(state2_).float()
        reward = game.get_reward()
        done = game.game_over()
        exp =  (state1, action_, reward, state2, done)
        replay.append(exp) #H
        state1 = state2
        
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            Q1 = model(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch) #B
            
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if j % sync_freq == 0: #C
                model2.load_state_dict(model.state_dict())
        if done or mov > max_moves:
            status = 0
            mov = 0
            
tf = time()
total_time = tf-t0
print(total_time)
#game.close()
losses = np.array(losses)

plt.figure(figsize=(10,7))
plt.plot(losses[-200:])
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)






