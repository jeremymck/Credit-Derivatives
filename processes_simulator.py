#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:28:46 2020

@author: jeremymarck
"""

# Importing Packages
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import norm
import scipy.stats as stats
from datetime import timedelta

######################
## Hawkes Simulator ##
######################

### 1) A function for generating a Hawkes process.
##################################################

# Parameters: 
# lambda_0 = initial intensity
# alpha and beta : Kernel parameters.
# T = time horizon.

def hawkes_process(lambda0, alpha, beta, T):
    # Initialisation de la liste des temps.
    t = []
    # Initialisation de l'intensité.
    lambd = lambda0
    # Premier évènement.
    n = 1
    u = np.random.random()
    s = -np.log(u)/lambd
    if s <= T:
        t.append(s)
    else:
        return t
    # Main loop.
    n += 1
    lambd = lambda0
    # Update maximum intensity.
    for j in range(len(alpha)):
        lambd += alpha[j]
    # H = différence d'intensités.
    H = lambd - lambda0
    # Nouveau tirage
    u = np.random.random()
    s -= np.log(u)/lambd
    # La boucle.
    while s <= T:
        # Tirage d'une uniforme entre 0 et 1.
        d = np.random.random()
        H1 = H
        # Application du noyau
        for j in range(len(alpha)):
            H1 = H1*np.exp(-beta[j]*(s - t[-1]))
        lambdaS = lambda0 + H1
        # Principale condition.
        if d <= lambdaS/lambd:
            t.append(s)
            n += 1
            lamb = lambdaS
            H = H1
            for j in range(len(alpha)):
                lambd += alpha[j]
                H += alpha[j]
        else:
            lambd = lambdaS
        u = np.random.random()
        s -= np.log(u)/lambd
    return t


### 2) Functions for implementing the Kernel and get the intensity.
###################################################################

def kernel1D(alpha,beta,t):
    return alpha*np.exp(-beta*t)

def lambdaa(lamb0,alpha,beta,t,s):
    somme = lamb0
    for i in range(len(t)):
        if s >= t[i]:
            somme += kernel1D(alpha,beta,s-t[i])
        else:
            break
    return somme


### 3) Simulation of a Hawkes process : plots.
##############################################

# Simulating a Hawkes process.
t = hawkes_process(1.2,[0.6],[0.8],10)
print(len(t))

# Getting the associated intensities.
x = np.linspace(0,10,100)
l_y = []
for i in range(len(x)):
    l_y.append(lambdaa(1.2,0.6,0.8,t,x[i]))
    
plt.plot(t, 'b:o')
plt.title('Hawkes Process - Events - lam0 = 1.2, alpha = 0.2, beta = 0.8, T = 10')
plt.xlabel('Number of events')
plt.ylabel('Time horizon')
plt.grid(True)
plt.show()

plt.plot(l_y, color = 'red')
plt.title('Intensities - Hawkes Process')
plt.ylabel('Intensity value')
plt.xlabel('Events')
plt.grid(True)
plt.show()


###################################
## Poisson Homogeneous simulator ##
###################################

def poisson_simulator(lamb, T):
    time_list = []
    # Initial setting.
    t = 0
    i = 1
    # First random draw.
    u = np.random.random()
    s = - (1/lamb)*np.log(u)
    while t + s <= T:
        time = t + s
        t = t + s
        i += 1
        time_list.append(time)
    return time_list

a = poisson_simulator(1.2, 10)
print(len(a))
plt.plot(a, 'b:o')
plt.title('Homogeneous Poisson process - Events - lambda = 1.2, T = 10')
plt.xlabel('Number of events')
plt.ylabel('Time horizon')
plt.grid(True)
plt.show()


        
