import Naive_Bayes as NB
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('banknote.csv', index_col=False)
#print(df.describe())

train_data, test_data = train_test_split(df , test_size = 0.2 , shuffle=True)
#print(test_data.head())

GNB = NB.GaussianNaiveBayes()
GNB.train(train_data)
GNB.test(test_data)
print('Accuracy:' , GNB.accuracy)