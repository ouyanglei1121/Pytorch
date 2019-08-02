#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/7/25
import numpy as np
def computer_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b))**2
    return totalError/float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)

    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")  # 读取文件data.csv中的数据，并以，号分割
    learning_raet = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 100
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_raet, num_iterations)
    print("b = {0} , w = {1} ".format(b, w))
    print("error = {0}".format(computer_error_for_line_given_points(b, w, points)))

if __name__=="__main__":
    run()

