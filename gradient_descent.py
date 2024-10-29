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
    theta = theta0
    change_theta = 1
    num_iter = 0

    # Start val: [ .5, -.2, 2.5]
    # Goal:      [.75,  .5,   1]
    while change_theta > tol and num_iter < max_iter:
      w1_sum = 0
      for gradient in gradient_funcs[0]:
        w1_sum += gradient(theta[0], theta[1], theta[2])
      w1 = theta[0] - alpha * .25 * w1_sum

      w2_sum = 0
      for gradient in gradient_funcs[1]:
        w2_sum += gradient(theta[0], theta[1], theta[2])
      w2 = theta[1] - alpha * .25 * w2_sum

      b_sum = 0
      for gradient in gradient_funcs[2]:
        b_sum += gradient(theta[0], theta[1], theta[2])
      b = theta[2] - alpha * .25 * b_sum

      change_theta = max(abs(w1 - theta0[0]), abs(w2 - theta0[1]), abs(b - theta0[2]))
      theta = [w1, w2, b]

      num_iter+=1
    
    thetas = [theta]
    return thetas

def minimize_gradient_descent_analytically(J_func,alpha,theta0,h,tol=1e-10,max_iter=500,debug=True):
    debug=False
    """
    You can stop optimizing when the max absolute change in theta is < tol.
    You can also stop when you reach max_iterations.
    Make sure you use the theta from the previous iteration for all your calculations until the next "epoch". In other words, make sure you are making copies correctly and not updated say w1 and then using it to help you update w2. The updates to the parameters should be done in parallel.
    """

    theta = theta0
    change_theta = 1
    num_iter = 0


    while change_theta > tol and num_iter < max_iter:
      J = J_func(theta[0], theta[1], theta[2])
      
      J_w1 = J_func(theta[0]+h, theta[1], theta[2])
      J_w2 = J_func(theta[0], theta[1]+h, theta[2])
      J_b = J_func(theta[0], theta[1], theta[2]+h)

      
      grad_w1 = (J_w1-J)/h
      grad_w2 = (J_w2-J)/h
      grad_b = (J_b-J)/h

      w1 = theta[0] - alpha * grad_w1
      w2 = theta[1] - alpha * grad_w2
      b = theta[2] - alpha * grad_b

      change_theta = max(abs(w1-theta[0]), abs(w2-theta[1]), abs(b-theta[2]))
      
      theta = [w1, w2, b]

      num_iter+=1

      if num_iter < 10 and debug==True:
        print(grad_w1)


    thetas = [theta]
    return thetas

