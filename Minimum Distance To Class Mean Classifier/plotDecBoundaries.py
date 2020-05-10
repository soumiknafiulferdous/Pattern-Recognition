import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries(training, label_train, sample_mean):
    nclass = max(np.unique(label_train))

    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    inc = 0.003

    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')))

    dist_mat = cdist(xy, sample_mean)
    pred_label = np.argmin(dist_mat, axis=1)
    decisionmap = pred_label.reshape(image_size, order='F')
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')
    plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'bo')

    if nclass == 2:
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
    else:
        l = plt.legend('Class 1', loc=2)
    plt.gca().add_artist(l)

    m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    if nclass == 2:
        l1 = plt.legend([m1,m2],['Class 1 Mean', 'Class 2 Mean'], loc=4)
    plt.gca().add_artist(l1)

    plt.show()






