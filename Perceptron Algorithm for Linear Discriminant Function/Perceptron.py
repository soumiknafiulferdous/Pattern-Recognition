import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

np.random.seed(0)

#task 1
df = pd.read_csv('train.txt', sep=" ", header=None, dtype='float64')

arr = df.values

w1x, w1y, w2x, w2y = [], [], [], []

for i in range(len(arr)):
    if arr[i][2] == 1.0:
        w1x.append(arr[i][0])
        w1y.append(arr[i][1])
    else:
        w2x.append(arr[i][0])
        w2y.append(arr[i][1])

y = np.empty((0,6), int)

#task 2
for i in range(len(arr)):
    if arr[i][2] == 1.0:
        x1_2 = arr[i][0]**2
        x2_2 = arr[i][1]**2
        x1_x2 = arr[i][0] * arr[i][1];
        y = np.append(y, np.array([[x1_2, x2_2, x1_x2, arr[i][0], arr[i][1], 1]]), 0)
    else:
        x1_2 = -(arr[i][0]**2)
        x2_2 = -(arr[i][1]**2)
        x1_x2 = -(arr[i][0] * arr[i][1]);
        y = np.append(y, np.array([[x1_2, x2_2, x1_x2, -arr[i][0], -arr[i][1], -1]]), 0)
#print(y)

#task 3_and_4
w_one = np.ones((1, 6),dtype=int)
w_zero = np.zeros((1, 6),dtype=int)
w_rand = np.random.random((1,6))

print("When w=1: ", w_one);
print("When w=0: ", w_zero);
print("When w=rand: ", w_rand);

alpha_list = []

one_at_a_time1 = [];
many_at_a_time1 = [];

one_at_a_time0 = [];
many_at_a_time0 = [];

one_at_a_time2 = [];
many_at_a_time2 = [];


#One_at_a_time
def one_at_a_time_func(w, alpha):
    check = 0;
    count = 0;

    while check != 6:
        count = count + 1;
        check = 0;
        for i in range(6):
            g = np.dot(y[i, :], w.transpose());
            if g <= 0:
                w = w + alpha * y[i, :];
            else:
                check = check + 1;
        if check == 6:
            break;
    return count;

#many_at_a_time
def many_at_a_time_func(w, alpha):
    check = 0;
    count = 0;
    w_new = 0;

    while check != 6:
        count = count + 1;
        check = 0;
        for i in range(6):
            g = np.dot(y[i, :], w.transpose());
            if g <= 0:
                w_new = w_new + y[i, :];
            else:
                check = check + 1;
        if check == 6:
            break;
        w = w + alpha * w_new;
    return count;


#for_1
for i in range(1,11,1):
    alpha_list.append(float(i / 10))
    one_at_a_time1.append(one_at_a_time_func(w_one,float(i/10)))
for i in range(1, 11, 1):
    many_at_a_time1.append(many_at_a_time_func(w_one, float(i / 10)))

#for 0
for i in range(1,11,1):
    one_at_a_time0.append(one_at_a_time_func(w_zero,float(i/10)))
for i in range(1, 11, 1):
    many_at_a_time0.append(many_at_a_time_func(w_zero, float(i / 10)))

#for random
for i in range(1,11,1):
    one_at_a_time2.append(one_at_a_time_func(w_rand,float(i/10)))
for i in range(1, 11, 1):
    many_at_a_time2.append(many_at_a_time_func(w_rand, float(i / 10)))

print("\nAlpha values: ",alpha_list)
print("\nFor 1:")
print("One at a time: w1 ",one_at_a_time1)
print("Many at a time: w1 ",many_at_a_time1)
print("\nFor 0:")
print("One at a time: w0 ",one_at_a_time0)
print("Many at a time: w0 ",many_at_a_time0)
print("\nFor Random:")
print("One at a time: wR ",one_at_a_time2)
print("Many at a time: wR ",many_at_a_time2)

plt.plot(w1x,w1y,'^r',label='Class 1')
plt.plot(w2x,w2y,'ob',label='Class 2')

plt.legend();
plt.show();


#Bar chart for 1
bar_width = 0.02
bar1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time1, bar_width,
color='b',
label='one at a time', align='center')

bar2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time1, bar_width,
color='r',
label='many at a time', align='center')

plt.xlabel('Learning rate')
plt.ylabel('Number of iterations')
plt.title('All weight one')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()

#Bar chart for 0
bar_width = 0.02
bar1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time0, bar_width,
color='b',
label='one at a time', align='center')

bar2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time0, bar_width,
color='r',
label='many at a time', align='center')

plt.xlabel('Learning rate')
plt.ylabel('Number of iterations')
plt.title('All weight zero')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()

#Bar chart for random
bar_width = 0.02
bar1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time2, bar_width,
color='b',
label='one at a time')

bar2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time2, bar_width,
color='r',
label='many at a time')

plt.xlabel('Learning rate')
plt.ylabel('Number of iterations')
plt.title('All weight Random')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()
