# Example using numpy and sklearn
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


RW_A1= 2
RW_A2= 3
RW_A3= 4
RW_A4= 5
RW_A5= 6
RW_B1= 7
RW_B2= 8
RW_B3= 9
RW_B4= 10
RW_B5= 11
RW_C1= 12
RW_C2= 13
RW_C3= 14
RW_C4= 15
RW_C5= 16
RW_D1= 17
RW_D2= 18
RW_D3= 19
RW_D4= 20
RW_D5= 21
FW_3= 3
FW_5= 5
FW_7= 7

# X, Y are your grid coordinates, Z is the value
X = np.array([[FW_3,RW_A1], [FW_3,RW_A2],[FW_3,RW_A3],[FW_3,RW_A4],[FW_3,RW_A5],[FW_3,RW_B1], [FW_3,RW_B2],[FW_3,RW_B3],[FW_3,RW_B4],[FW_3,RW_B5],[FW_3,RW_C1], [FW_3,RW_C2],[FW_3,RW_C3],[FW_3,RW_C4],[FW_3,RW_C5],[FW_3,RW_D1], [FW_3,RW_D2],[FW_3,RW_D3],[FW_3,RW_D4],[FW_3,RW_D5], [FW_5,RW_A1], [FW_5,RW_A2],[FW_5,RW_A3],[FW_5,RW_A4],[FW_5,RW_A5],[FW_5,RW_B1], [FW_5,RW_B2],[FW_5,RW_B3],[FW_5,RW_B4],[FW_5,RW_B5],[FW_5,RW_C1], [FW_5,RW_C2],[FW_5,RW_C3],[FW_5,RW_C4],[FW_5,RW_C5],[FW_5,RW_D1], [FW_5,RW_D2],[FW_5,RW_D3],[FW_5,RW_D4],[FW_5,RW_D5], [FW_7,RW_A1], [FW_7,RW_A2],[FW_7,RW_A3],[FW_7,RW_A4],[FW_7,RW_A5],[FW_7,RW_B1], [FW_7,RW_B2],[FW_7,RW_B3],[FW_7,RW_B4],[FW_7,RW_B5],[FW_7,RW_C1], [FW_7,RW_C2],[FW_7,RW_C3],[FW_7,RW_C4],[FW_7,RW_C5],[FW_7,RW_D1], [FW_7,RW_D2],[FW_7,RW_D3],[FW_7,RW_D4],[FW_7,RW_D5]])
Z = np.array([0.823, 0.839, 0.855, 0.871, 0.886, 0.901, 0.915, 0.928, 0.941, 0.954, 0.966, 0.977, 0.988, 0.999, 1.009, 1.019, 1.028, 1.036, 1.044, 1.051, 0.858, 0.874, 0.890, 0.906, 0.921, 0.935, 0.949, 0.963, 0.976, 0.988, 1.000 , 1.012, 1.023, 1.034, 1.044, 1.053, 1.062, 1.071, 1.079, 1.087, 0.868, 0.885, 0.901, 0.916, 0.931, 0.946, 0.960, 0.973, 0.986, 0.999, 1.011, 1.022, 1.034, 1.044, 1.054, 1.064, 1.073, 1.081, 1.090, 1.097])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, Z)
r2 = model.score(X_poly, Z)
# model.coef_ and model.intercept_ give you the polynomial coefficients

# ...existing code...

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R² score:", r2)