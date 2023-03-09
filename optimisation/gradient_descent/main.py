from plot import basic_line_plot, gradient_descent_line_plot
import quadratic_function
import saddle_point_function
from gradient_descent import gradient_descent



def run_quadratic_function():
    print('Running quadratic function')

    basic_line_plot(quadratic_function.function,
                    'quadratic function',
                    x_range={'lower': -6, 'upper': 10})

    history, result = gradient_descent(start=9,
                                       gradient=quadratic_function.gradient_function,
                                       learn_rate=0.1,
                                       max_iter=1000)

    gradient_descent_line_plot(function=quadratic_function.function,
                               step_history=history,
                               result=result,
                               image_name='quadratic function gradient descent line plot',
                               x_range={'lower': -6, 'upper': 10})

    print(history)
    print(result)


def run_saddle_point_function():
    print('Running saddle point function')

    basic_line_plot(saddle_point_function.function, 
                    'Saddle point function',
                    x_range={'lower': -1, 'upper': 3})

    history, result = gradient_descent(start=-0.5,
                                       gradient=saddle_point_function.gradient_function,
                                       learn_rate=0.3,
                                       max_iter=1000)

    gradient_descent_line_plot(function=saddle_point_function.function,
                               step_history=history,
                               result=result,
                               image_name='Saddle point gradient descent line plot',
                               x_range={'lower': -1, 'upper': 2})

    print(history)
    print(result)


if __name__ == '__main__':
    run_quadratic_function()
    run_saddle_point_function()
