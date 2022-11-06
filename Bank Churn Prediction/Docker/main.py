#!flask/bin/python

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to scale the data using z-score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Library to split data set into train and test sets
from sklearn.model_selection import train_test_split

# Library to plot confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve
from sklearn import metrics

# Library to import keras backend
from tensorflow.keras import backend, optimizers
from keras import callbacks
# Library to avoid the warnings
import warnings
warnings.filterwarnings("ignore")

# importing different functions to build NN models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# Importing Tensorflow library
import tensorflow as tf

# Loading csv data into dataframe
filename = 'Churn.csv'
df = pd.read_csv(filename)

# Dropping columns identified earlier
df.drop(['RowNumber','CustomerId','CreditScore_cat'],axis=1,inplace=True)
# Split Data into target and independent variables
X = df.drop('Exited',axis=1)
y = df['Exited']
# One Hot Encode our categorical features. 
X = pd.get_dummies(X, columns=['Surname','Geography','Gender'], drop_first=True)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1,stratify = y)

# Fixing the seed 
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# Initializing model
model = Sequential()
# Adding input layer with 64 nuerons, relu activation function, he_unifrom the weight initializer
model.add(Dense(activation = 'relu', input_dim = X_train.shape[1],kernel_initializer='he_uniform', units=64))
# Add droput after input layer
model.add(Dropout(0.9))
# Adding hiddenlayer with 32 nuerons, relu activation func, he_uniform
model.add(Dense(activation = 'relu',kernel_initializer='he_uniform', units=32))
model.add(Dropout(0.9))
# Adding output layer 
model.add(Dense(1, activation = 'sigmoid')) 

# Defining optimizer with Adam
opt = optimizers.Adam(lr=0.001)

# Compile model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Fit model on train and validation with 50 epochs
model.fit(X_train, y_train,           
          validation_split=0.2,
          epochs=50,
          verbose=1)

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


