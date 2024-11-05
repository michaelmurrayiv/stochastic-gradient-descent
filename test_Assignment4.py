import sys
import os
sys.path.append("..")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
import joblib 

answers = joblib.load("./answers_Assignment4.joblib")

import gradient_descent
import torch
import pandas as pd
import numpy as np

gradients_w1 = [lambda w1,w2,b: 0, lambda w1,w2,b: 0, lambda w1,w2,b: 2*(w1+b-1.75), lambda w1,w2,b: 2*(w1+w2+b-2.25)]
gradients_w2 = [lambda w1,w2,b: 0, lambda w1,w2,b: 2*(w2+b-1.5), lambda w1,w2,b: 0, lambda w1,w2,b: 2*(w1+w2+b-2.25)]
gradients_b = [lambda w1,w2,b: 2*(b-1), lambda w1,w2,b: 2*(w2+b-1.5), lambda w1,w2,b: 2*(w2+b-1.5), lambda w1,w2,b: 2*(w1+w2+b-2.25)]

F1_func = lambda w1,w2,b: (b-1)**2
F2_func = lambda w1,w2,b: (w2+b-1.5)**2
F3_func = lambda w1,w2,b: (w1+b-1.75)**2
F4_func = lambda w1,w2,b: (w1+w2+b-2.25)**2
R_func = lambda w1,w2,b: 1/4*(F1_func(w1,w2,b)+F2_func(w1,w2,b)+F3_func(w1,w2,b)+F4_func(w1,w2,b))
F_funcs = [F1_func, F2_func, F3_func, F4_func]

def test_1():    
    solution_thetas = answers['minimize_gradient_descent']
    answer_thetas = gradient_descent.minimize_gradient_descent([gradients_w1,gradients_w2,gradients_b],0.1,[0.5,-0.2,2.5])
    assert np.all(np.abs(np.array(solution_thetas)-np.array(answer_thetas)) <= 0.0001)
    
def test_2():    
    solution_thetas = answers['minimize_gradient_descent_analytically']
    answer_thetas = gradient_descent.minimize_gradient_descent_analytically(R_func,0.1,[0.5,-0.2,2.5],0.01)
    assert np.all(np.abs(np.array(solution_thetas)-np.array(answer_thetas)) <= 0.0001)

# test that SGD values reach expected values
def test_3():    
    solution_thetas = answers['minimize_gradient_descent']
    answer_thetas = gradient_descent.minimize_stochastic_gradient_descent([gradients_w1,gradients_w2,gradients_b],0.1,[0.5,-0.2,2.5])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001

# test that SGD Analytical values reach expected values
def test_4():    
    solution_thetas = answers['minimize_gradient_descent_analytically']
    answer_thetas = gradient_descent.minimize_stochastic_gradient_descent_analytically(F_funcs,0.1,[0.5,-0.2,2.5],0.01)
    print(answer_thetas[-1])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001

# SGD with momentum
def test_5():    
    solution_thetas = answers['minimize_gradient_descent']
    answer_thetas = gradient_descent.minimize_sgd_momentum([gradients_w1,gradients_w2,gradients_b],0.1,[0.5,-0.2,2.5])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001

# SGD Analytical with momentum
def test_6():    
    solution_thetas = answers['minimize_gradient_descent_analytically']
    answer_thetas = gradient_descent.minimize_sgd_momentum_analytically(F_funcs,0.1,[0.5,-0.2,2.5],0.01)
    print(answer_thetas[-1])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001

# Adam
def test_7():    
    solution_thetas = answers['minimize_gradient_descent']
    answer_thetas = gradient_descent.minimize_sgd_adam([gradients_w1,gradients_w2,gradients_b],0.1,[0.5,-0.2,2.5])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001

# Adam Analytical
def test_8():    
    solution_thetas = answers['minimize_gradient_descent_analytically']
    answer_thetas = gradient_descent.minimize_sgd_adam_analytically(F_funcs,0.1,[0.5,-0.2,2.5],0.01)
    print(answer_thetas[-1])
    for i in range(len(solution_thetas[-1])):
        assert abs(solution_thetas[-1][i] - answer_thetas[-1][i]) <= 0.0001
