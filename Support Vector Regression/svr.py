#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api


#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
y=y.reshape(-1,1)


#feature scaling
from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
x=sx.fit_transform(x)
sy=StandardScaler()
y=sy.fit_transform(y)

#SVR model
from sklearn.svm import SVR
reg= SVR(kernel='rbf')
reg.fit(x,y)

#prediction of results
y_pred= sy.inverse_transform(reg.predict(sx.fit_transform(np.array([[6.5]]))))


#visualization
plt.scatter(x,y)
plt.plot(x,reg.predict(x), color='red')
plt.xlabel('level')
plt.ylabel('salary')
plt.title('truth or bluff')
