#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Salary_Data.csv')
x= dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting of the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=1/3, random_state=0)


#linear regression (training set)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#predicting the test results
y_pred=reg.predict(x_test)

#visualization of training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, reg.predict(x_train), color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#visualization of testing set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, reg.predict(x_train), color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()