import math

class GaussianNaiveBayes:

    def __init__(self):
        self.prediction = []
        self.accuracy = 0

    def train(self,train_data):
        train_data_label_zero = train_data[ train_data['y'] == 0]
        # drop the label column
        train_data_label_zero = train_data_label_zero.drop('y', axis = 1)
        #print(train_data_label_zero.head())

        train_data_label_one = train_data[ train_data['y'] == 1]
        # drop the label column
        train_data_label_one = train_data_label_one.drop('y', axis = 1)
        #print(train_data_label_one.head())

        # Calculate standard deviation
        self.train_zero_standard_deviation = train_data_label_zero.std()
        self.train_one_standard_deviation = train_data_label_one.std()

        # Calculate the sum
        self.train_zero_mean = train_data_label_zero.mean()
        self.train_one_mean = train_data_label_one.mean()
        #print(train_data_label_one.mean())


    def test(self, test_data):

        # reset the index so that it can be accessed with a loop
        test_data.reset_index(inplace=True)

        # Drop the test label
        test_data_without_label = test_data.drop('y' , axis = 1)

        # Get the column names
        columns = list(test_data_without_label.columns.values)
        columns = columns[1:]

        for i in range(0 , len(test_data_without_label)):
            self.probability_of_zero = 1
            self.probability_of_one = 1

            # iterating for each column to calculate probability
            for column in columns:
                self.probability_of_zero *= self.Calculate_Probability(float(test_data.loc[i, column]),
                                                                       float(self.train_zero_mean[column]),
                                                                       float(self.train_zero_standard_deviation[column]))
                self.probability_of_one *= self.Calculate_Probability(float(test_data.loc[i ,column]),
                                                                       float(self.train_one_mean[column]),
                                                                       float(self.train_one_standard_deviation[column]))

            if self.probability_of_one > self.probability_of_zero:
                self.prediction.append(1)
            else:
                self.prediction.append(0)

        error = 0

        for i in range(0 , len(test_data)):
            if int(test_data.loc[i , 'y']) != self.prediction[i]:
                error += 1

        self.accuracy = (error/len(test_data)) * 100

        true_negative=0
        false_positive=0
        true_positive=0
        false_negative=0

        for i in range(len(self.prediction)):
            if(int(test_data.iloc[i]['y'])==0 and self.prediction[i]==0):
                true_negative += 1
            elif(int(test_data.iloc[i]['y'])==0 and self.prediction[i]==1):
                false_positive += 1
            elif(int(test_data.iloc[i]['y'])==1 and self.prediction[i]==1):
                true_positive += 1
            elif(int(test_data.iloc[i]['y'])==1 and self.prediction[i]==0):
                false_negative += 1

        print("True Negative:"+str(true_negative))
        print("True Positive:"+str(true_positive))
        print("False Negative:"+str(false_negative))
        print("False Positive:"+str(false_positive))

    def Calculate_Probability(self, x , mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent