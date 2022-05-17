# -*- coding: utf-8 -*-

##Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# https://github.com/Ranadheer-senani/AI-MachineLearningLab/tree/main/Datasets-lab
df = pd.read_csv('https://raw.githubusercontent.com/Ranadheer-senani/AI-MachineLearningLab/main/Datasets-lab/linear-house.csv')
df.head(5)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.xlabel('Area in SFT')
plt.ylabel('Price in USD')
plt.scatter(df.area,df.price,color='red',marker='*')

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

reg.predict([[3300]])

reg.coef_

reg.intercept_

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.xlabel('Area in SFT')
plt.ylabel('Price in USD')
plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')

testdf = pd.read_csv('https://github.com/Ranadheer-senani/AI-MachineLearningLab/raw/main/Datasets-lab/areas.csv')
testdf

testdf['pred-prices'] = list(map(int,reg.predict(testdf[['area']])))
testdf

"""##Multiple Linear Regression"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('https://github.com/Ranadheer-senani/AI-MachineLearningLab/raw/main/Datasets-lab/houseprice-multi.csv')
df

df.bedrooms.median()

median_beds = int(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_beds)
df

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

reg.predict([[3000,3,40]])
reg.coef_
reg.intercept_

"""## Logistic Regression"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv("https://github.com/Ranadheer-senani/AI-MachineLearningLab/raw/main/Datasets-lab/insurance-logistic.csv")
df.head(5)

plt.scatter(df.age,df.bought_ins,marker = "+", color="red")

x_train,x_test,y_train,y_test = train_test_split(df[['age']],df.bought_ins,test_size=0.1)

x_test

model = LogisticRegression()
model.fit(x_train,y_train)

model.predict(x_test)

x_test

model.score(x_test,y_test)

model.predict_proba(x_test)

sns.regplot(x=df.age,y=df.bought_ins,data=df,logistic=True,ci=None)

"""##Decision Tree"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv('https://raw.githubusercontent.com/Ranadheer-senani/AI-MachineLearningLab/main/Datasets-lab/salaries-decisiontree.csv')
df

inputs = df.drop('salary_more_then_100k', axis = 'columns')
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs

inputs_n = inputs.drop(['company','job','degree'],axis='columns')

inputs_n

model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target)

model.score(inputs_n,target)

#Is salary of a Google Computer Engineer with a Bacheolers Degree > 100k ?
model.predict([[2,1,0]])

#Is salary of a Google Computer Programmer with a masters degree > 100k ?
model.predict([[2,1,1]])

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=['company_n','job_n','degree_n'],  
                   class_names='salary_more_then_100k',
                   filled=True)

"""## Reinforcement Learning

"""

import random
from typing import List

class SampleEnvironment:
  def __init__(self):
    self.steps_left = 20

  def get_observation(self) -> List[float]:
    return[0.0,0.0,0.0]
  
  def get_actions(self) -> List[int]:
    return [0,1]

  def is_done(self)-> bool:
    return self.steps_left == 0

  def action(self, action:int) -> float:
    if self.is_done():
      raise Exception("Game is over")
    self.steps_left -=1
    return random.random()

class Agent:
  def __init__(self):
    self.total_reward = 0.0

  def step(self, env: SampleEnvironment):
    current_obs = env.get_observation()
    print(current_obs)
    actions = env.get_actions()
    print(actions)
    reward = env.action(random.choice(actions))
    self.total_reward +=reward
    print("Total reward {}".format(self.total_reward))

if __name__ == "__main__":
  env = SampleEnvironment()
  agent = Agent()
  i=0
  while not env.is_done():
    i = i+1
    print("Steps {}".format(i))
    agent.step(env)
    print("Total reward got: ", agent.total_reward)

"""## Artificial Neural Network with Back Propagation"""

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.5, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            if k % 10000 == 0: print ('epochs:', k)
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T,np.array(x)))      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 0])

    nn.fit(X, y)

    for e in X:
        print(e,nn.predict(e))



"""##SVM"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://github.com/Ranadheer-senani/AI-MachineLearningLab/raw/main/Datasets-lab/SVM.csv')
df.head()

x=df.iloc[:,:2]
y=df.Y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape,y_train.shape)

model = SVC(kernel="rbf")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)

"""# Supervised Learning"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
Model = KNeighborsClassifier(n_neighbors=6)
Model.fit(iris.data,iris.target)
X = [
      [5.9, 1.0, 5.1, 1.8],
      [3.4, 2.0, 1.1, 4.8],
    ]
Model.predict(X)