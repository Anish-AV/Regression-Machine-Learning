#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


#import dataset
dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values



#regression
from sklearn.tree import DecisionTreeRegressor
reg= DecisionTreeRegressor(random_state=0)
reg.fit(x,y)


#prediction
y_pred=reg.predict(np.array([[6.5]]))

#visualization
plt.scatter(x,y, color= 'red')
plt.plot(x,reg.predict(x),color= 'green')
plt.xlabel('position')
plt.ylabel('salary')
plt.title('decision tree regression')



x_grid= np.arange(min(x), max(x), 0.01 )
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color= 'red')
plt.plot(x_grid,reg.predict(x_grid),color= 'blue')
plt.xlabel('position')
plt.ylabel('salary')
plt.title('decision tree regression')