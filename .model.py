from sklearn.cluster import KMeans
import pandas as pd
#from load import read_dataset 

# Load the dataset
df = pd.read_csv("C:\\Users\\Karim\\Desktop\\bd-a1\\diabetes.csv")

# Select suitable columns for K-means
# Let's assume 'Glucose' and 'BMI' are the columns selected for clustering
X = df[['Glucose', 'BMI']]

# Apply K-means algorithm with k=3 and n_init=10
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Save the number of records in each cluster
cluster_counts = df['cluster'].value_counts()
cluster_counts.to_csv('k.txt', header=False)

#read_dataset()