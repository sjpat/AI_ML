#!flask/bin/python

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist, pdist

# to perform clustering and compute scoring
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


file = 'Credit Card Customer Data.xlsx'
data = pd.read_excel(file)
df = data.copy()

# First step for feature engineering will be dropping the two columns identified earlier
df.drop(['Sl_No', 'Customer Key'], axis=1, inplace=True)

# Next we'll scale our features to improve model performance
std_scaler = StandardScaler()

df_scaled = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
df_scaled.head()

# Initialize KMeans model for 3 clusters
model = KMeans(n_clusters=3, random_state=1)
# Fit on scaled dataset
model.fit(df_scaled)
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


