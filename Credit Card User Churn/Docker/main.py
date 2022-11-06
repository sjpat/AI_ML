#!flask/bin/python
# Library to suppress warnings or deprecation notes 
import warnings
warnings.filterwarnings('ignore')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
# Libraries to split data, impute missing values 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Library to import Logistic Regression
from sklearn.linear_model import LogisticRegression 

# Libraries to import decision tree classifier and different ensemble classifiers
from sklearn.ensemble import GradientBoostingClassifier

# Remove limits for the number of displayed columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# Library for SHAP
import shap

# Import libraries to create custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('BankChurners.csv')


# Encoding our target variable (Attrition_Flag) with 1 for attrited customer and 0 for existing customer
encode = {'Attrited Customer': 1, 'Existing Customer': 0}

df['Attrition_Flag'].replace(encode, inplace=True)

# Creating a copy of the data to build the model
df1 = df.copy()

# Separating target and dependent features
# Dependent features
X = df1.drop('Attrition_Flag',axis=1)
X = pd.get_dummies(X)

# Target feature
y = df1['Attrition_Flag']


# Split our data into train, val and test sets

# First splitting our data set into a temp and test set
X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)

# Now we're splitting our temporary set into train and val

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1, stratify=y_temp
)

# Defining my columns that need log transformation 
cols_to_log = ['Customer_Age', 'Months_on_book', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Transform Training data
for colname in cols_to_log:
    X_train[colname + '_log'] = np.log(X_train[colname] + 1)
X_train.drop(cols_to_log, axis=1, inplace=True)

# Transform validation data
for colname in cols_to_log:
    X_val[colname + '_log'] = np.log(X_val[colname] + 1)
X_val.drop(cols_to_log, axis=1, inplace=True)

# Transform test data
for colname in cols_to_log:
  X_test[colname+'_log'] = np.log(X_test[colname]+1)
X_test.drop(cols_to_log, axis=1, inplace=True)


# For this dataset, we have three categorical features with missing values so I will employ a simple imputer to replace with the most frequent
imputer = SimpleImputer(strategy='most_frequent')
impture = imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Build model with best parameters
model = GradientBoostingClassifier(
    random_state=1,
    subsample=0.9,
    n_estimators=250,
    min_samples_split=10,
    max_features= 0.9
)

# Fit model to training data
model.fit(X_train, y_train)
pickle.dump(model, open('model.pkl', 'wb'))

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    data = request.get_json()  # Get data posted as a json
    data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])

if __name__ == '__main__':
    if os.environ['ENVIRONMENT'] == 'production':
        app.run(port=80,host='0.0.0.0')


