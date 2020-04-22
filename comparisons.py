#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:07:12 2020

@author: jeremymarck
"""

from scipy import stats
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from random import gauss
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy.random as npr


## Chi-square discretization.
#############################

def y_cir_path(y0,k,mu,sigma,n_simul,M,T):
    df = (4*mu*k)/(sigma**2)
    dt = T/M
    l_path = []
    for i in range(n_simul):
        path = [y0]
        for t in range(1,M+1):
            C = (sigma**2*(1 - math.exp(-k*dt)))/(4*k)
            arg = (4*k*math.exp(-k*dt)*path[-1])/(sigma**2*(1 - math.exp(-k*dt)))
            chi2 = npr.noncentral_chisquare(df, arg)
            S = C*chi2
            path.append(S)
        l_path.append(path)
    return l_path

#### Comparing 100 simulations for different risk levels:
#########################################################
#low_chi = y_cir_path(0.00001, 0.9, 0.0001,0.01, 100, 500,2)
#middle = y_cir_path(0.01, 0.8, 0.02,0.2, 100, 500,2) 
#high = y_cir_path(0.03, 0.5, 0.05,0.5, 100, 500,2) 

#### Plots:
###########
    
#for i in range(len(low)):
  #plt.plot(low[i])
  #plt.title('Low risk levels - Chi-square - 100 simulations - 2 years')
  #plt.grid(True)
  #plt.xlabel('Time steps')
  #plt.ylabel('CIR process values')
#plt.savefig('/Users/jeremymarck/Desktop/Chi_Low.pdf')
  
#for i in range(len(middle)):
  #plt.plot(middle[i])
  #plt.title('Middle risk levels - Chi2 - 100 simulations - 2 years')
  #plt.grid(True)
  #plt.xlabel('Time steps')
  #plt.ylabel('CIR process values')
#plt.savefig('/Users/jeremymarck/Desktop/Middle_Chi2.pdf')
  
#for i in range(len(high)):
  #plt.plot(high[i])
  #plt.title('High risk levels - Chi 2 - 100 simulations - 2 years')
  #plt.grid(True)
  #plt.xlabel('Time steps')
  #plt.ylabel('CIR process values')
#plt.savefig('/Users/jeremymarck/Desktop/High_Chi2.pdf')


## Euler-Scheme.
################

def euler_path(y0,k,mu,sigma,n_simul,M,T):
    dt = T/M
    l_path = []
    for i in range(n_simul):
        path = [y0]
        for j in range(M):
            z = gauss(0,1)
            y = path[-1] + k*(mu-path[-1])*dt + sigma*np.sqrt(path[-1]*dt)*z
            path.append(y)
        l_path.append(path)
    return l_path

#low_e = euler_path(0.00001, 0.9, 0.0001,0.01, 100, 500,2) 
#middle_e = euler_path(0.01, 0.8, 0.02,0.2, 100, 500,2) 
#high_e = euler_path(0.03, 0.5, 0.05,0.5, 100, 500,2) 

#### Plots:
###########
    
#for i in range(len(low_e)):
  #plt.plot(low_e[i])
  #plt.title('Low risk levels - Euler - 100 simulations - 2 years')
  #plt.grid(True)
  #plt.xlabel('Time steps')
  #plt.ylabel('CIR process values')
#plt.savefig('/Users/jeremymarck/Desktop/low_euler.pdf')
  
#for i in range(len(middle_e)):
  #plt.plot(middle_e[i])
  #plt.title('Middle risk levels - Euler - 100 simulations - 2 years')
  #plt.grid(True)
  #plt.xlabel('Time steps')
  #plt.ylabel('CIR process values')
#plt.savefig('/Users/jeremymarck/Desktop/Middle_Euler.pdf')
  
        
## Milstein-Scheme.
##################
def milstein_path(y0,k,mu,sigma,n_simul,M,T):
    dt = T/M
    l_path = []
    for i in range(n_simul):
        path = [y0]
        for j in range(M):
            z = gauss(0,1)
            y = path[-1] + k*(mu-path[-1])*dt + sigma*np.sqrt(path[-1]*dt)*z + 0.25*sigma*sigma*((np.sqrt(dt)*z)**2 - dt)            
            path.append(y)
        l_path.append(path)
    return l_path

low_m = milstein_path(0.00001, 0.9, 0.0001,0.01, 100, 500,2) 
middle_m = euler_path(0.01, 0.8, 0.02,0.2, 100, 500,2) 
high_m = milstein_path(0.03, 0.5, 0.05,0.5, 100, 500,2) 

for i in range(len(low_m)):
    plt.plot(low_m[i])
    plt.grid(True)
    plt.xlabel('Time steps')
    plt.ylabel('CIR process values')
    plt.title('Low risk - Milstein - 100 simulations - 2 years')
plt.savefig('/Users/jeremymarck/Desktop/Low_M.pdf')

for i in range(len(middle_m)):
    plt.plot(middle_m[i])
    plt.grid(True)
    plt.xlabel('Time steps')
    plt.ylabel('CIR process values')
    plt.title('Middle risk - Milstein - 100 simulations - 2 years')
plt.savefig('/Users/jeremymarck/Desktop/Middle_M.pdf')

for i in range(len(high_m)):
    plt.plot(high_m[i])
    plt.grid(True)
    plt.xlabel('Time steps')
    plt.ylabel('CIR process values')
    plt.title('High risk - Milstein - 100 simulations - 2 years')
plt.savefig('/Users/jeremymarck/Desktop/High_M.pdf')
