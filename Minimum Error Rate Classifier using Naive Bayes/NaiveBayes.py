import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sigma1 = np.array([[.25, .3], [.3, 1.0]])
sigma2 = np.array([[.5, 0], [0, .5]])

mean1 = np.array([0, 0])
mean2 = np.array([2, 2])

df = pd.read_csv('test.txt', sep=",", header=None)
arr = df.values

x = np.empty((0, 2), int)

for i in range(len(arr)):
    x = np.append(x, np.array([[arr[i][0], arr[i][1]]]), 0)

n1_value = []
n2_value = []

def norm_dist(x, mean, sd, w):
    m = 2;
    sd_inv = np.linalg.inv(sd)
    xu_norm = (x - mean)
    p1 = sd_inv.dot(np.transpose(xu_norm))
    f1 = 0.5 * (xu_norm.dot(p1))
    f2 = (m / 2) * np.log(2 * np.pi)
    sd_det = np.linalg.det(sd)
    f3 = 0.5 * np.log(sd_det)
    total = -f1 - f2 - f3
    return np.exp(total + np.log(w))

for i in range(len(x)):
    value = norm_dist(x[i], mean1, sigma1, 0.5)
    n1_value.append(value)

for i in range(len(x)):
    value = norm_dist(x[i], mean2, sigma2, 0.5)
    n2_value.append(value)

class1_valueX = []
class1_valueY = []
class2_valueX = []
class2_valueY = []

for i in range(len(arr)):
    if n1_value[i] > n2_value[i]:
        class1_valueX.append(arr[i][0])
        class1_valueY.append(arr[i][1])
    elif n2_value[i] > n1_value[i]:
        class2_valueX.append(arr[i][0])
        class2_valueY.append(arr[i][1])

print("N1: ", n1_value)
print("N2: ", n2_value)

# print("value separating: ")
# print("class1 x", class1_valueX)
# print("class1 y", class1_valueY)
# print("class2 x", class2_valueX)
# print("class2 y", class2_valueY)

class1_valueZs = np.zeros(len(class1_valueX))
class2_valueZs = np.zeros(len(class2_valueX))

X5 = np.arange(np.min(arr[:, 0]), np.max(arr[:, 0]), 0.05)
Y5 = np.arange(np.min(arr[:, 1]), np.max(arr[:, 1]), 0.05)

X5, Y5 = np.meshgrid(X5, Y5)

pos5 = np.empty(X5.shape + (2,))
print("Postion: ", X5.shape)
pos5[:, :, 0] = X5
pos5[:, :, 1] = Y5

def m_g(pos, mean, sd):
    n = mean.shape[0]
    sd_det = np.linalg.det(sd)
    sd_inv = np.linalg.inv(sd)
    N = np.sqrt(np.power(2 * np.pi, n) * sd_det)
    fac = np.einsum('ijk,kl,ijl->ij', pos - mean, sd_inv, pos - mean)
    return np.exp(-fac / 2) / N

Z1 = m_g(pos5, mean1, sigma1)
Z2 = m_g(pos5, mean2, sigma2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X5, Y5, Z1, rstride=8, cstride=8, alpha=0.4, cmap=cm.ocean)
cset = ax.contour(X5, Y5, Z1, zdir='z', offset=-0.18, cmap=cm.ocean)

ax.plot_surface(X5, Y5, Z2, rstride=8, cstride=8, alpha=0.4, cmap=cm.ocean)
cset = ax.contour(X5, Y5, Z2, zdir='z', offset=-0.17, cmap=cm.ocean)

Z3 = Z1 - Z2;
cset = ax.contour(X5, Y5, Z3, zdir='z', offset=-0.17, cmap=cm.ocean)

ax.scatter(class1_valueX, class1_valueY, class1_valueZs, c='r', marker='*')
ax.scatter(class2_valueX, class2_valueY, class2_valueZs, c='b', marker='o')

ax.set_zlabel("Probability Density")
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-0.15, 0.5)
ax.view_init(30, -125)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Implementing Minimum Error Rate Classifier")
plt.show()
