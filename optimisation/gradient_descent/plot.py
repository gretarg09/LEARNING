from matplotlib import pyplot as plt
import numpy as np


def basic_line_plot(function, image_name, x_range):
    fig, ax = plt.subplots()

    x = np.linspace(x_range['lower'], x_range['upper'], 10000)
    ax.plot(x, function(x))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{image_name}')

    plt.savefig(F'images/{image_name}.png')



def gradient_descent_line_plot(function, step_history, result, image_name, x_range):
    fig, ax = plt.subplots()

    x = np.linspace(x_range['lower'], x_range['upper'], 10000)
    ax.plot(x, function(x))
    ax.scatter(step_history, function(np.array(step_history)), color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{image_name}')

    plt.savefig(F'images/{image_name}.png')
