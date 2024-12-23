import torch
import random
import math
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
            val = random.randint(0, len(variable_funcs)-1) # choose random gradient function
            gradient = variable_funcs[val](*thetas[-1])
            theta[i] = thetas[-1][i] - alpha * gradient
            i += 1
        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)
      
        thetas += [theta]
        num_iter += 1
    
    return thetas

def minimize_stochastic_gradient_descent_analytically(F_funcs, alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
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
        val = random.randint(0, len(F_funcs)-1) # choose random gradient function
        start_val = F_funcs[val](*theta)
        for i in range(len(theta)):
            theta_copy = thetas[-1][:]
            theta_copy[i] += h
            
            J_h = F_funcs[val](*theta_copy)
            grad = (J_h - start_val)/h
            
            theta[i] = theta[i] - alpha * grad

        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)

        thetas += [theta]

        num_iter += 1

        if num_iter < 10 and debug==True:
            print(grad_w1)

    return thetas

def minimize_sgd_momentum(gradient_funcs,alpha,theta0,tol=1e-10,max_iter=500):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    # theta0 = w1, w2, b
    change_theta = 1
    num_iter = 0    
    thetas = [theta0]

    beta = .9 # coefficient for momentum
    velocity = [0,0,0] # initialize Vt
    
    # Start val: [ .5, -.2, 2.5]
    # Goal:      [.75,  .5,   1]
    while change_theta > tol and num_iter < max_iter:
        theta = [0,0,0]
        i = 0
        for variable_funcs in gradient_funcs:
            val = random.randint(0, len(variable_funcs)-1) # choose random gradient function
            gradient = variable_funcs[val](*thetas[-1])

            velocity[i] = -alpha * gradient + velocity[i] * beta
            
            theta[i] = thetas[-1][i] + velocity[i]
            
            i += 1
            
        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)
      
        thetas += [theta]
        num_iter += 1
    
    return thetas

def minimize_sgd_momentum_analytically(F_funcs, alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    debug=False
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    change_theta = 1
    num_iter = 0 
    thetas = [theta0]
    beta = .9
    velocity = [0,0,0]

    while change_theta > tol and num_iter < max_iter:
        theta = thetas[-1][:]
        val = random.randint(0, len(F_funcs)-1) # choose random gradient function
        start_val = F_funcs[val](*theta)
        for i in range(len(theta)):
            theta_copy = thetas[-1][:]
            theta_copy[i] += h
            
            J_h = F_funcs[val](*theta_copy)
            grad = (J_h - start_val)/h

            change = -alpha * grad + beta * velocity[i]
            
            theta[i] = theta[i] + change

        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)

        thetas += [theta]

        num_iter += 1

        if num_iter < 10 and debug==True:
            print(grad_w1)

    return thetas

def minimize_sgd_adam(gradient_funcs,alpha,theta0,tol=1e-10,max_iter=500):
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """
    
    # theta0 = w1, w2, b
    change_theta = 1
    num_iter = 0    
    thetas = [theta0]

    epsilon = 1e-8
    beta1 = .9 # hyperparameters for Adam
    beta2 = .999
    velocity1 = [0,0,0] # initialize velocities
    velocity2 = [0,0,0]
    
    # Start val: [ .5, -.2, 2.5]
    # Goal:      [.75,  .5,   1]
    while change_theta > tol and num_iter < max_iter:
        theta = [0,0,0]
        i = 0
        for variable_funcs in gradient_funcs:
            val = random.randint(0, len(variable_funcs)-1) # choose random gradient function
            gradient = variable_funcs[val](*thetas[-1])
            
            # Adam formulas
            velocity1[i] = velocity1[i] * beta1 + gradient * (1 - beta1) 
            velocity2[i] = velocity2[i] * beta2 + gradient**2 * (1 - beta2)

            theta[i] = thetas[-1][i] - alpha * (velocity1[i] / (math.sqrt(velocity2[i]) + epsilon))
            
            i += 1
            
        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)
      
        thetas += [theta]
        num_iter += 1
    
    return thetas

def minimize_sgd_adam_analytically(F_funcs, alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    debug=False
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    change_theta = 1
    num_iter = 0 
    thetas = [theta0]
                
    epsilon = 1e-8
    beta1 = .9 # hyperparameters for Adam
    beta2 = .999
    velocity1 = [0,0,0] # initialize velocities
    velocity2 = [0,0,0]
    
    while change_theta > tol and num_iter < max_iter:
        theta = thetas[-1][:]
        val = random.randint(0, len(F_funcs)-1) # choose random gradient function
        start_val = F_funcs[val](*theta)
        for i in range(len(theta)):
            theta_copy = thetas[-1][:]
            theta_copy[i] += h
            
            J_h = F_funcs[val](*theta_copy)
            grad = (J_h - start_val)/h

            velocity1[i] = velocity1[i] * beta1 + grad * (1 - beta1) 
            velocity2[i] = velocity2[i] * beta2 + grad**2 * (1 - beta2)

            theta[i] = theta[i] - alpha * (velocity1[i] / (math.sqrt(velocity2[i]) + epsilon))

        changes = [abs(theta[i] - thetas[-1][i]) for i in range(len(theta))]
        change_theta = max(changes)

        thetas += [theta]

        num_iter += 1

        if num_iter < 10 and debug==True:
            print(grad_w1)

    return thetas