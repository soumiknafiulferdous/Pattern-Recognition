# Pattern Recognition

These classifiers are mainly used for data visualization, manipulation and plotting. Several datasets are used for training and testing the models. Data are plotted in the graph to figure out the result of testing in comparison with training with the help of these algorithms.

## Naive Bayes Algorithm 
Data is predicted successfully using the naive bayes algorithm.

## Minimum Distance to Class Mean Classifier
It can classify the test dataset in it's desired classes according to the train dataset considering the minimum distance.

At first, two-class set of prototypes have to be taken from 'train.txt' and 'test.txt' files. Then plot all sample points (train data) from both classes, but samples from the same class should have the same color and marker. Then, using a minimum distance classifier with respect to class-mean, classify the test data points by plotting them with the designated class-color but different marker. Use the Linear Discriminant Function for that. After that, draw the decision boundary between the two-classes and find accuracy. Legends can be added to the graph but legends should not cover the actual graph.

## Perceptron Algorithm
The Perceptron algorithm is implemented for finding the weights of a Linear Discriminant function. This algorithm is used for batch processing (many at a time) and single processing (one at a time) and the result is showed in a graph.

At first, take input from 'train.txt' file and plot all sample points from both classes. Samples
from the same class should have the same color and marker. Then, observe if these two classes can be separated with a linear boundary. Consider the case of a second order polynomial discriminant function. After that, generate the high dimensional sample points y and also normalize any one of the two classes. Use Perceptron Algorithm (both one at a time and many at a time) for finding the weight-coefficients of the discriminant function (i.e., values of w) boundary for the linear classifier. Here, α is the learning rate and 0 < α ≤ 1. Three initial weights have to be used (all one, all zero, randomly initialized with seed fixed). For all of these three cases vary the learning rate between 0.1 and 1 with step size 0.1. Create a table which should contain the learning rate, number of iterations for one at a time and batch Perceptron for all of the three initial weights. Finally, create a bar chart for visualizing the table data.

## Minimum Error Rate Classifier

A minimum error rate classifier for a two-class problem has been designed considering they follow the normal distribution. At first, classify the sample points from 'test.txt' and the classified samples should have different colored markers according to the assigned class label. Then draw a figure which should include the corresponding probability distribution function along with its contour. At last, draw the decision boundary. Here, library function is not used for calculating values from normal distribution. For classified samples, different colored markers are used according to the assigned class label. 

## K-Nearest Neighbors (KNN) 

At first, determine parameter K = number of nearest neighbors and calculate the distance between the query-instance and all the training samples. Then, sort the distance and determine nearest neighbors based on the K-th minimum distance. After that, gather the category of the nearest neighbors and use simple majority of the category of nearest neighbors as the prediction value of the query instance. Then the dataset is classified accurately based on this algorithm where user can input the number of nearest neighbors and the dataset is classified according to that number.

## K-Means Clustering

For  K-means clustering, determine number of cluster K at first and assume the centroid or center of these clusters. Then take any random objects as the initial centroids or the first K objects in sequence that can also serve as the initial centroids. After that, plot all the points. Perform the k-means clustering algorithm applying Euclidean distance as a distance measure on the given dataset with k=2. Color the corresponding points on the clusters with different colors. Finally, the dataset is clustered successfully according to K-means clustering algorithm.
