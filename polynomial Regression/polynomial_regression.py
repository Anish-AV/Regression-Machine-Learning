#import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values


#linear reg
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(x,y)

#polynomial reg
from sklearn.preprocessing import PolynomialFeatures
preg=PolynomialFeatures(degree=4)
x_p=preg.fit_transform(x)
lreg1=LinearRegression()
lreg1.fit(x_p,y)

#visualizing linear reg
plt.scatter(x,y, color='blue')
plt.plot(x, lreg.predict(x), color='red')
plt.title('level vs salary')
plt.xlabel('level')
plt.ylabel('salary')

#visualizing polynomial reg
x_grid=np.arange(min(x),max(x), 0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='blue')
plt.plot(x_grid, lreg1.predict(preg.fit_transform(x_grid)), color='red')
plt.title('level vs salary')
plt.xlabel('level')
plt.ylabel('salary')

a=[[6.5]]

#predict linear
lreg.predict(a)

#predict polynomial
lreg1.predict(preg.fit_transform(a))
