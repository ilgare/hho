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
# This is the variant NHO proposed to replace the original Harris' Hawks
# algorithm, with extra code to track its history to generate Figures
# 2 and 3 in the article.

import numpy as np
import random
from numpy.random import random_sample as rand
from math import gamma as gamma


class NHO:
    def __init__(self, obj_func, N, T, lb, ub, dim):
        self.obj_func = obj_func
        self.N = N                      # Nuber of hawks
        self.T = T                      # number of iterations
        self.ub = np.array([ub]).flatten()  # lower bound
        self.lb = np.array([lb]).flatten()  # lower bound
        self.dim = dim                  # dimension of data vectors
        self.hawks = np.zeros((N, dim))

        self.history = np.zeros((T,N,dim))

        self.rabbit_history = np.zeros((T,dim))

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
        Scale = self.ub - self.lb

        Power1 = 2
        Power12 = 4
        C1 = 10
        C12 = 0.02
        Power2 = 3
        Power22 = 0
        C2 = 2 # Bunlar kucuk olmamali?
        C22 = 4

        CNVG = np.zeros(self.T)

        for t in range(0, self.T):

            self.history[t] = self.hawks.copy()

            # Check boundaries
            self.hawks = np.maximum(np.minimum(self.hawks, self.ub), self.lb)

            for i in range(0, self.N):
                fitness = self.obj_func(self.hawks[i])
                if fitness < Rabbit_Energy:
                    Rabbit_Energy = fitness
                    Rabbit_Location = self.hawks[i].copy()
                    self.rabbit_history[t] = Rabbit_Location.copy()                    

            # factor to show the decreaing energy of rabbit
            E1 = 2*(1 -((t/self.T)**1))

            # Update the location of Harris' hawks
            for i in range(0, self.N):
                E0 = 2*rand() - 1  # -1<E0<1
                Escaping_Energy = E1*(E0)  # escaping energy of rabbit

                if abs(Escaping_Energy) >= 1:

                    self.hawks[i] = self.lb + Scale*rand(self.dim)
                        #cases[1] += 1

                elif abs(Escaping_Energy) < 1:

                    r = rand()  # probablity of each event
                    rand_Hawk_index = np.random.randint(self.N)
                    hawk_rand = self.hawks[rand_Hawk_index]

                    r1 = rand()
                    
                    if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                        self.hawks[i] = self.hawks[i] + 10 * rand() * (Escaping_Energy ** 2) * ((r1*hawk_rand + (1-r1)* Rabbit_Location) - self.hawks[i])
                        #cases[2] += 1
                    
                    elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        hawk1 = Rabbit_Location + 10 * rand() * (Escaping_Energy ** 2) * (r1*self.hawks[i] + (1-r1)* hawk_rand) 
                        #cases[3] += 1
                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            hawk2 = Rabbit_Location + (2*rand(self.dim)-1)* 0.02 *  (Escaping_Energy ** 4) * (hawk_rand - hawk1)

                            # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2
                    
                    elif r >= 0.5 and abs(Escaping_Energy) < 0.5:
                        self.hawks[i] = Rabbit_Location + 2 * rand()* (abs(Escaping_Energy) ** 3) * (self.hawks[i] - Rabbit_Location) 
                        #cases[4] += 1

                    else:
                        hawk1 = Rabbit_Location + 2 * rand() * (abs(Escaping_Energy) ** 3) * (r1*self.hawks[i] + (1-r1)* hawk_rand) 
                            
                        #cases[5] += 1

                        # improved move?
                        if self.obj_func(hawk1) < self.obj_func(self.hawks[i]):
                            self.hawks[i] = hawk1
                        else:
                            hawk2 = Rabbit_Location + (2*rand(self.dim)-1)* 4 * (hawk_rand - hawk1)                             # improved move?
                            if self.obj_func(hawk2) < self.obj_func(self.hawks[i]):
                                self.hawks[i] = hawk2

            CNVG[t] = Rabbit_Energy

        return Rabbit_Energy, Rabbit_Location, CNVG, self.history, self.rabbit_history

