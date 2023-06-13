import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Create an array of values between 0.01 and 10 to represent p.
    # We start from 0.01 to avoid trying to take the logarithm of 0, which is undefined.
    p = np.linspace(0.01, 10, 100)

    # Compute the natural logarithm of p.
    y = np.log(p)

    # Create the plot.
    plt.figure(figsize=(8, 6))
    plt.plot(p, y)

    # Add title and labels.
    plt.title("Graph of y = log(p)")
    plt.xlabel("p")
    plt.ylabel("y")

   # Display the plot.
    plt.show()
