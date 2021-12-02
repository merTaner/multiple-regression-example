import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = LinearRegression()
regr.fit(X, y)
test_y = regr.predict(X)

# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2_1 = regr.predict([[2300, 1300]])

'''
We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, the CO2 emission will be approximately 107g.
What if we increase the weight with 1000kg?
'''
predictedCO2 = regr.predict([[3300, 1300]])
print("old value : ", predictedCO2_1)
print("new value : ", predictedCO2)

