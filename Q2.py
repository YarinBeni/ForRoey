# Q2a:
#In this code, we calculate:
# 1)Total Advertising Spend per advertiser by summing the 'spend' column for each advertiser.
# 2)Customer Acquisition Cost (CAC) per advertiser by dividing the total spend
#   by the total number of new customers for each advertiser.
# 3)Customer Lifetime Value (CLV) per advertiser by calculating the product of the total
#   number of transactions, average purchase value, and average customer lifespan for each advertiser.

import pandas as pd

# assuming this is my DataFrame:
df = pd.DataFrame({
    'advertiser': ['adv1', 'adv1', 'adv2', 'adv2', 'adv3', 'adv3'],
    'spend': [100, 200, 300, 400, 500, 600],
    'new_customers': [2, 2, 3, 3, 4, 4],
    'total_transactions': [2, 4, 3, 5, 6, 8],
    'purchase_value': [50, 100, 150, 200, 250, 300],
    'customer_lifespan': [1, 2, 1, 2, 3, 3]
})

def calculate_kpis(df):
    # Total Advertising Spend
    total_spend = df.groupby('advertiser')['spend'].sum()
    print(f"Total Advertising Spend:\n{total_spend}\n")

    # Customer Acquisition Cost (CAC)
    cac = df.groupby('advertiser').apply(lambda x: x['spend'].sum() / x['new_customers'].sum())
    print(f"Customer Acquisition Cost (CAC):\n{cac}\n")

    # Customer Lifetime Value (CLV)
    clv = df.groupby('advertiser').apply(
        lambda x: (x['total_transactions'].sum() * x['purchase_value'].mean() * x['customer_lifespan'].mean()))
    print(f"Customer Lifetime Value (CLV):\n{clv}\n")

calculate_kpis(df)


# Q2b:
# Below is a sample implementation for each of the requested regression models using a hypothetical dataset "data.csv".
# the models are: linear Regression, Polynomial Regression, Random Forest Regression , 
#                 XGboost, Neural Net Methods, Support Vector Machine(SVM)

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

