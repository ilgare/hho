# Code for the paper
#
# Reconsidering the Harris' Hawks Optimization Algorithm
#
# by
#
# Kemal Ilgar Eroglu
#
# 11/17/2024
#
# Author: Kemal Ilgar Eroğlu
#
# An randomly modified implementation of the Harris' Hawks Optimization 
# algorithm.The original algorithm code is directly ported from the 
# Matlab code in Ali Asghar Heidari's official repository
#
# https://github.com/aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications/tree/master
#
# Certain comments were copied "as is", along with their grammar errors.
#
#
# The main paper is
# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems,
# DOI: https://doi.org/10.1016/j.future.2019.02.028
#

import numpy as np
import random
from numpy.random import random_sample as rand
from math import gamma as gamma


def Levy(dim):
    beta = 1.5
    sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    return u/np.abs(v)**(1/beta)

class RHO:
    def __init__(self, obj_func, N, T, lb, ub, dim):
        self.obj_func = obj_func
        self.N = N                      # Nuber of hawks
        self.T = T                      # number of iterations
        self.ub = np.array([ub]).flatten()  # lower bound
        self.lb = np.array([lb]).flatten()  # lower bound
        self.dim = dim                  # dimension of data vectors
        self.hawks = np.zeros((N, dim))

        self.initialize_hawks()

    def initialize_hawks(self):

        if self.ub.shape[0] == 1:
            self.hawks = rand((self.N, self.dim)) * \
                (self.ub - self.lb) + self.lb
        else:
            for i in range(0, dim):
                self.hawks[:, i] = rand(
                    (N, 1)) * (self.ub[i] - self.lb[i]) + self.lb[i]

    def optimize(self):

        Rabbit_Location = np.zeros((1, self.dim))
        Rabbit_Energy = float('inf')

        CNVG = np.zeros(self.T)

        for t in range(0, self.T):
            # Check boundaries
            self.hawks = np.maximum(np.minimum(self.hawks, self.ub), self.lb)

            for i in range(0, self.N):
                fitness = self.obj_func(self.hawks[i])
                if fitness < Rabbit_Energy:
                    Rabbit_Energy = fitness
                    Rabbit_Location = self.hawks[i].copy()

            # factor to show the decreaing energy of rabbit
            E1 = 2*(1 -(t/self.T))

            # Update the location of Harris' hawks
            for i in range(0, self.N):
                E0 = 2*rand() - 1  # -1<E0<1
                Escaping_Energy = E1*(E0)  # escaping energy of rabbit

                if abs(Escaping_Energy) >= 1:
                    # Exploration: Harris' hawks perch randomly
                    # based on 2 strategy:

                    q = rand()
                    rand_Hawk_index = np.random.randint(self.N)
                    hawk_rand = self.hawks[rand_Hawk_index]
                    if q >= 0.5:
                        # perch based on other family members
                        #self.hawks[i] = hawk_rand + rand() * \
                        #    np.abs(hawk_rand + 5*rand() * self.hawks[i])
                        self.hawks[i] = hawk_rand+ 2.5* rand()*(self.hawks[i]-hawk_rand)
                        #self.hawks[i] = hawk_rand + rand() * (self.hawks[i] - hawk_rand)
                        #cases[0] += 1

                    else:
                        # perch on a random tall tree (random
                        # site inside group's home range)
                        #self.hawks[i] = (Rabbit_Location - np.mean(self.hawks, axis=0)) + \
                        #rand()*((self.ub-self.lb)*rand() + self.lb)
                        self.hawks[i] = (self.ub-self.lb)*rand() + self.lb
                        
                        #cases[1] += 1

                elif abs(Escaping_Energy) < 1:
                    # Exploitation:
                    # Attacking the rabbit using 4 strategies regarding
                    # the behavior of the rabbit

                    # phase 1: surprise pounce (seven kills)
                    # surprise pounce (seven kills): multiple, short rapid
                    # dives by different hawks

                    r = rand()  # probablity of each event

                    if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege
                        self.hawks[i] = Rabbit_Location + Escaping_Energy * rand()*(self.hawks[i] - Rabbit_Location) 
                            
                        #cases[4] += 1

                    if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege
                        # random jump strength of the rabbit
                        Jump_strength = 2*(1-rand())
                        self.hawks[i] = (Rabbit_Location )-Escaping_Energy*abs(Jump_strength*Rabbit_Location+self.hawks[i])
                        
                        #cases[2] += 1

                    # phase 2: performing team rapid dives (leapfrog movements)
                    # Soft besiege % rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        Jump_strength = 2*(1-rand())
                        hawk1 = Rabbit_Location - Escaping_Energy * abs(0.8*Jump_strength*Rabbit_Location+ 0.2*self.hawks[i])
                            
                        #cases[3] += 1

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location + (1-Jump_strength)*np.mean(
                                self.hawks, axis=0))+rand(self.dim)*Levy(self.dim)
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

                    # Hard besiege % rabbit try to escape by many zigzag deceptive motions
                    if r < 0.5 and abs(Escaping_Energy) < 0.5:
                        # hawks try to decrease their average location with the rabbit
                        Jump_strength = 2*(1-rand())
                        hawk1 = Rabbit_Location+0.5*Escaping_Energy * abs(Jump_strength*Rabbit_Location - np.mean(self.hawks, axis=0))
                            
                        #cases[5] += 1

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            # Perform levy-based short rapid dives
                            # around the rabbit
                            hawk2 = Rabbit_Location+0.5*Escaping_Energy*abs(Jump_strength*Rabbit_Location-np.mean(
                                self.hawks, axis=0))+Escaping_Energy*rand(self.dim)*Levy(self.dim)
                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

            CNVG[t] = Rabbit_Energy

        return Rabbit_Energy, Rabbit_Location, CNVG

