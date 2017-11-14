import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def visualize_kmeans(kmeans, data, show=True, background_img=None, x_min=0, x_max=640, y_min=0, y_max=640):
    # Visualize the results on kmeans
    # see: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

    plt.scatter(data[:, 1], data[:, 0], c=[cm.spectral(float(i) / 10) for i in kmeans.labels_],marker = '+', s = 4);

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 0],
                marker='x', s=169, linewidths=3,
                color='r', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if background_img is not None:
        plt.imshow(background_img)

    plt.gca().invert_yaxis()
    plt.xticks(())
    plt.yticks(())

    if show:
        plt.show()

    return plt

