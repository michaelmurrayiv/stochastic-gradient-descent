import torch

import numpy as np
import copy

def minimize_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10,max_iter=500):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    # theta0 = w1, w2, b
    change_theta = 1
    num_iter = 0    
    thetas = [theta0]

    # Start val: [ .5, -.2, 2.5]
    # Goal:      [.75,  .5,   1]
    while change_theta > tol and num_iter < max_iter:
        theta = [0,0,0]
        i = 0
        for variable_funcs in gradient_funcs:
            sum = 0
            for gradient in variable_funcs:
                sum += gradient(*thetas[-1])
            theta[i] = thetas[-1][i] - alpha * .25 * sum
            i += 1
        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)
      
        thetas += [theta]
        num_iter += 1
    
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    debug=False
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """


    change_theta = 1
    num_iter = 0 
    thetas = [theta0]

    while change_theta > tol and num_iter < max_iter:
        theta = thetas[-1][:]
        start_val = J_func(*theta)
        for i in range(len(theta)):
            theta_copy = thetas[-1][:]
            theta_copy[i] += h
            
            J_h = J_func(*theta_copy)
            grad = (J_h - start_val)/h
            
            theta[i] = theta[i] - alpha * grad

        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)

        thetas += [theta]

        num_iter += 1

        if num_iter < 10 and debug==True:
            print(grad_w1)

    return thetas
    
def minimize_stochastic_gradient_descent(gradient_funcs,alpha,theta0,tol=1e-10,max_iter=500):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    # theta0 = w1, w2, b
    change_theta = 1
    num_iter = 0    
    thetas = [theta0]

    # Start val: [ .5, -.2, 2.5]
    # Goal:      [.75,  .5,   1]
    while change_theta > tol and num_iter < max_iter:
        theta = [0,0,0]
        i = 0
        for variable_funcs in gradient_funcs:
            sum = 0
            for gradient in variable_funcs:
                sum += gradient(*thetas[-1])
            theta[i] = thetas[-1][i] - alpha * .25 * sum
            i += 1
        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)
      
        thetas += [theta]
        num_iter += 1
    
    return thetas

def minimize_stochastic_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    debug=False
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """


    change_theta = 1
    num_iter = 0 
    thetas = [theta0]

    while change_theta > tol and num_iter < max_iter:
        theta = thetas[-1][:]
        start_val = J_func(*theta)
        for i in range(len(theta)):
            theta_copy = thetas[-1][:]
            theta_copy[i] += h
            
            J_h = J_func(*theta_copy)
            grad = (J_h - start_val)/h
            
            theta[i] = theta[i] - alpha * grad

        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)

        thetas += [theta]

        num_iter += 1

        if num_iter < 10 and debug==True:
            print(grad_w1)

    return thetas