#import lib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


#import the dataset
dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values



#regression
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=1000, random_state=0)
reg.fit(x,y)

#prediction
y_pred=reg.predict(np.array([[6.5]]))

#visualization
x_grid=np.arange(min(x), max(x), 0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid, reg.predict(x_grid), color='blue')
plt.xlabel("position")
plt.ylabel("salaries")
plt.title("Random Forests")