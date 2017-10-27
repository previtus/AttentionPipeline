import matplotlib.pyplot as plt
import numpy as np

def visualize_time_measurements(time_measurements, names, title, show=True, save=False, y_min=0.0, y_max=0.1, save_path=''):
    # Visualize the results on kmeans
    # see: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

    for time_series in time_measurements:
        plt.plot(time_series)

    data_y_max = y_max
    for t in time_measurements:
        m = np.max(t)
        data_y_max = np.max([m,data_y_max])

    plt.ylim(y_min, data_y_max)

    #plt.gca().invert_yaxis()
    #plt.xticks(())
    #plt.yticks(())
    plt.title(title)

    plt.ylabel('time (s)')
    plt.xlabel('image')

    plt.legend(names, loc='best')

    if show:
        plt.show()

    if save:
        plt.savefig(save_path)

    plt.clf()
    return plt
