import numpy as np
import torch


# 计算总误差
def comput_err_for_line_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    # 求每一个点的梯度，并累加，求平均（批量梯度下降）
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 循环更新参数
def gradiend_descent_runner(points, starting_b, starting_m,
                            laerning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), laerning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("Ldata.csv", delimiter=",")
    learning_rate = 0.0001
    ini_b = 0
    ini_w = 0
    num_iterations = 1000
    print("Starting gradient decent at b={0},w={1},error={2}"
          .format(ini_b, ini_w,
                  comput_err_for_line_points(ini_b, ini_w, points)))
    print("Running...")
    [b, m] = gradiend_descent_runner(points, ini_b, ini_w, learning_rate, num_iterations)
    print("After {0} iterations b={1},m={2},error={3}"
          .format(num_iterations, b, m, comput_err_for_line_points(b, m, points)))


if __name__ == '__main__':
    run()
