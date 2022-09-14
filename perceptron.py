import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class Perceptron:

    def __init__(self,learningRate = 0.01,epochs = 1000,random_state = 1) -> None:

        self.learningRate = learningRate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self,X,Y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.float_(0.)
        
        self.errors_ = []



        for i in range(self.epochs):
            errors=0
            print("epoch",i,'************')
            for x,y in zip(X,Y):

                change = self.learningRate * (y -self.predict(x))
                self.w = self.w + change * x
                self.b = self.b + change
                errors += int(change != 0.0)
                print(self.w)
            self.errors_.append(errors)
            print(self.errors_)
                

        return self 


    def predict(self,X):
        result = np.dot(X, self.w) + self.b
        return np.where(result>=0,1,0)



try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')
    
df.tail()



y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(0.1,10)
ppn.fit(X, y)
y_pred = ppn.predict(X)

print(accuracy_score(y, y_pred))


