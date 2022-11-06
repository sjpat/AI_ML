#!flask/bin/python

import warnings

warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data

import pandas as pd
import numpy as np

# Library to split data
from sklearn.model_selection import train_test_split

# Normalizing using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

# Selecting pandas plotting backend 
# pd.options.plotting.backend = "hvplot"

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)


# To build model for prediction
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# To work with categorical features that have order
from sklearn.preprocessing import OrdinalEncoder

# Plot ROC Curve
from sklearn.metrics import plot_roc_curve

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

file = 'Loan_Modelling.csv'
data = pd.read_csv(file)
df = data.copy()

# Transforming columns identified above 
cols_to_log = ["Income", "CCAvg", "Mortgage"]
for colname in cols_to_log:
    df[colname + "_log"] = np.log(df[colname] + 1)
df.drop(cols_to_log, axis=1, inplace=True)

# Updating num_cols with the transformed columns
num_cols = ['Age', 'Experience', 'Income_log', 'Family', 'CCAvg_log', 'Mortgage_log']
#Normalizing num_cols
df[num_cols] = MinMaxScaler().fit_transform(
    df[num_cols]
)

# Split data
X = df.drop(['ZIPCode', 'Personal_Loan', 'lat', 'lng', ], axis=1)
Y = df["Personal_Loan"]

X = pd.get_dummies(X, drop_first=True)

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)

clf = DecisionTreeClassifier(random_state=1)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train a decision tree using the effective alphas
clfs = []
for alpha in ccp_alphas:
    clf=DecisionTreeClassifier(random_state=1, ccp_alpha = alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Creating model with best train and test recall
index_best_model = -3
model = clfs[index_best_model]
    
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


