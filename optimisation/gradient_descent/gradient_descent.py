
import numpy as np

def gradient_descent(start,
                     gradient,
                     learn_rate,
                     max_iter,
                     tol=0.01):
    '''
    1. Choose a starting point (initialisation)
    2. Calculate gradient at this point
    3. Make a scaled step in the opposite direction to the gradient (objective: minimise)
        - p(n+1) = p(n) - learning_rate * gradient(p(n))
    4. Repeat points 2 and 3 until one of the criteria is met:
    5. Maximum number of iterations reached
    6. Step size is smaller than the tolerance (due to scaling or a small gradient).

    Parameters:
    -----------
    start: the starting point, in our case we define it manually but in practice it is often
          a random initialisation.
    gradient: the gradient function
    learn_rate: the learning rate
    max_iter: the maximum number of iterations
    tol: the tolerance to conditionally stop the algorithm (in this case a default value is 0.01).
    '''


    steps = [start] # history tracking
    x = start

    for _ in range(max_iter):

        print(f'the gradient is {gradient(x)}')

        diff = learn_rate * gradient(x)

        if np.abs(diff) < tol:
            break

        x = x - diff

        steps.append(x) # history tracing

    return steps, x
