import matplotlib.pyplot as plt
import numpy as np

def visualize_time_measurements(time_measurements, names, show=True, x_min=0, x_max=640, y_min=0, y_max=640):
    # Visualize the results on kmeans
    # see: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

    for time_series in time_measurements:
        plt.plot(time_series)

    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)

    #plt.gca().invert_yaxis()
    #plt.xticks(())
    #plt.yticks(())
    plt.title("Time measurements")

    plt.ylabel('time (s)')
    plt.xlabel('image')

    plt.legend(names, loc='best')

    if show:
        plt.show()

    return plt

