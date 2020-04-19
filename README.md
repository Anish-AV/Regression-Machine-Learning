# Regression 

Regression using Machine Learning models like Linear Regression, 
Polynomial Regression, Support Vector Regression, 
Decision Tree and Random Forest

## Libraries installation
scikit Learn
```bash
pip install -U scikit-learn
```
Pandas
```bash
pip install pandas
```
Numpy
```bash
pip install numpy
```

## Models

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
```

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
```
### Support Vector Regression
```python
from sklearn.svm import SVR
```
### Decision Tree
```python
from sklearn.tree import DecisionTreeRegressor
```
### Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
```
