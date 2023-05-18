import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data.csv')
X = df['X'].values.reshape(-1,1)
y = df['y'].values
# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression ~~~~~~~~~~~~~~~~~~~
from sklearn.linear_model import LinearRegression
# Initialize and fit the model  and predict
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2. Polynomial Regression ~~~~~~~~~~~~~~~
from sklearn.preprocessing import PolynomialFeatures
# Create Polynomial Features
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
# Train-Test split
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
# Initialize and fit the model and predict
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_train, y_poly_train)
y_pred = lin_reg_2.predict(X_poly_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3. Random Forest Regression ~~~~~~~~~~~~
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
# Initialize and fit the model and predict
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 4. XGBoost ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                          max_depth = 5, alpha = 10, n_estimators = 1000)
xg_reg.fit(X_train, y_train)
# Predict
y_pred_xg = xg_reg.predict(X_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 5. Neural Network Methods ~~~~~~~~~~~~~~
from sklearn.neural_network import MLPRegressor
# Initialize and fit the model and predict
nn_reg = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', learning_rate='adaptive',
                      max_iter=1000, learning_rate_init=0.01, alpha=0.01)
nn_reg.fit(X_train, y_train)
y_pred = nn_reg.predict(X_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 6. Support Vector Machine (SVM) ~~~~~~~
from sklearn.svm import SVR
# Initialize and fit the model and predict
svm_reg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svm_reg.fit(X_train, y_train)
y_pred = svm_reg.predict(X_test)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

