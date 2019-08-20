#Task 18: Machine Learning V
#Polynomial regression: Salary and years of experience

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Training set
x_train = [[1.1], [2.0], [2.9], [3.2], [4.1], [5.3], [6.0], [7.1], [8.2], [9.5], [10.5]]#Years of experience
y_train = [[39343], [43525], [56642], [54445], [57081], [83088], [93940], [98273], [113812], [116969], [121872]]#Salary

#Testing set
x_test = [[1.5], [2.2], [3.2], [4.5], [5.9], [6.8], [9.0], [10.3]]#Years of experience
y_test = [[37731], [39891], [64445], [61111], [81363], [91738], [105582], [122391]] #Salary

#Linear Regression model to plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 25, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

#Degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=5)

#ThE preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

#Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

#Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='y', linestyle='--')
plt.title('Salary regressed on diameter')
plt.xlabel('Diameter in Years')
plt.ylabel('Salary')
plt.axis([0, 12, 0, 140000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print(x_train)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)


