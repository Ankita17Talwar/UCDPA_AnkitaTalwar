# Group stocks based on financial metrics : ROE, ROA, beta
# We will use K-Mean Clustering : Unsupervised ML algorithm
# finds groups/clusters in the data, with the number of groups represented by the variable 'k'(hence the name).
# Inputs to the K-means algorithm are the data/features(Xis) and the value of 'K'(number of clusters to be formed).

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from yellowbrick.cluster import SilhouetteVisualizer

###################################################################################
# 1. Read data from csv - NIFTY 500 Stock Data (This csv is output of Stock_ESG_price.py
###################################################################################
# data = pd.read_csv('../output_files_500_nifty/Stock_Info_All_Feilds.csv', index_col=0)
data = pd.read_csv('../output_files/out_esg_info/Stock_Info_yFin_Data.csv', index_col=0)

###################################################################################
# 2. Choose Metrics : ROE, ROA, dividendYield
###################################################################################
df_data = data[['returnOnEquity', 'returnOnAssets', 'dividendYield', 'symbol']].copy()

###################################################################################
# 3. Create sector data frame and  Set index - symbol;
# Set index of our metrics data frame and Drop Null Data
###################################################################################
sector_df = data[['symbol', 'sector']].copy()
sector_df.set_index('symbol', inplace=True)

# Remove any zeros
df_data = df_data[df_data[['returnOnEquity', 'returnOnAssets', 'dividendYield', 'symbol']] != 0]
print(df_data.shape)

print('Drop NaN')
# Drop Null data
df_data.dropna(how='any', inplace=True)

print(df_data.shape)

df_data.set_index('symbol', inplace=True)
print(df_data.shape)

# Convert to percentages and round to two decimal places
df_data['returnOnEquity'] = (df_data['returnOnEquity'] * 100).round(2)
df_data['returnOnAssets'] = (df_data['returnOnAssets'] * 100).round(2)
df_data['dividendYield'] = (df_data['dividendYield'] * 100).round(2)

print(df_data.head())

df_describe = df_data.describe()

df_describe.loc['+3_std'] = df_describe.loc['mean'] + (df_describe.loc['std'] * 3)
df_describe.loc['-3_std'] = df_describe.loc['mean'] - (df_describe.loc['std'] * 3)

# print(df_data.info())
print(df_describe)
# making a copy of DataFrame
df = df_data.copy()
###################################################################################
# 4. Visualize Original data
###################################################################################

fig = plt.figure()
ax = Axes3D(fig)

# define x,y,z for plot
x = list(df.iloc[:, 0])
y = list(df.iloc[:, 1])
z = list(df.iloc[:, 2])

# Set axes labels
column_names = df.columns
ax.set_xlabel(column_names[0] + '%')
ax.set_ylabel(column_names[1] + '%')
ax.set_zlabel(column_names[2] + '%')
ax.scatter(x, y, z, c='green', marker='v')
ax.text2D(0.05, 0.95, "Initial Data Plot", transform=ax.transAxes)
plt.show()
fig.savefig('plots/clustering/OrignalData_ScatterPlt.png')

###################################################################################
# 5 To Standardize Data We use Robust Scaler :
# Standard scaler is not suitable as data may have outliers and hence Standardization
# can be skewed.To overcome this, the median and interquartile range can be used
# when standardizing numerical input variables, generally referred to as robust scaling.
###################################################################################

robust_scaler = RobustScaler()

########################################################
# 6. Visualize Scaled Data
########################################################

X_train_robust = robust_scaler.fit_transform(df.values)
fig = plt.figure()
ax = Axes3D(fig)

# take the scaled data in this example.
x = X_train_robust[:, 0]
y = X_train_robust[:, 1]
z = X_train_robust[:, 2]

# define the axes labels
column_names = df.columns
ax.set_xlabel(column_names[0] + '%')
ax.set_ylabel(column_names[1] + '%')
ax.set_zlabel(column_names[2] + '%')

# create a new plot
ax.scatter(x, y, z, c='red')

ax.text2D(0.05, 0.95, "Scaled Data Plot", transform=ax.transAxes)

plt.show()
fig.savefig('plots/clustering/ScaledData_ScatterPlt.png')
###############################################################################################
# 7. Create K-Means Model with two cluster
# Here we will use pipeline to scale and fit the model
####################################################################################################

km_model = KMeans(n_clusters=2, random_state=1200)

pipeline = make_pipeline(robust_scaler, km_model)
pipeline.fit(df.values)
labels = pipeline.predict(df.values)

df['cluster'] = labels

###############################################################################################
# 8. Plot Clustered Data.
###############################################################################################

# print(df.head())
print(df.columns)
# print(df['returnOnEquity'].values)

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig)

# axis
x = df['returnOnEquity'].values
y = df['returnOnAssets'].values
z = df['dividendYield'].values
#c = df['cluster'].values
#c = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

colors = np.random.randint(224, size=(224))
# define the axes labels
column_names = df.columns
ax.set_xlabel(column_names[0])
ax.set_ylabel(column_names[1])
ax.set_zlabel(column_names[2])


# create a new plot
ax.scatter(x, y, z, c=df['cluster'],  cmap='winter')
ax.text2D(0.05, 0.95, "CLUSTERS from k-means algorithm with k = 2", transform=ax.transAxes)

centroids = pipeline.named_steps['kmeans'].cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
centroids_z = centroids[:, 2]
print(centroids_z)
print(centroids_y)
ax.scatter(centroids_x, centroids_y,centroids_z, marker='D', c='r')
plt.show()
fig.savefig('plots/clustering/Kmean_2_Plt.png')

sect_cluster = df.merge(sector_df, how='left', on='symbol')
sect_cluster.to_csv('../output_files/initial_cluster.csv')

print(sect_cluster.head())

###############################################################################################
# 9. Elbow Method to choose value of K
# We randomly used value of k =2 while fitting the model. One of the way to choose best value of k
# is to check model's inertia  i.e. distance of points in a cluster from its centroid
# As more and more clusters are added, the inertia keeps on decreasing,
# creating what is called an 'elbow curve'. We select the value of k beyond which we do not see
# much benefit (i.e., decrement) in the value of inertia.
# calculating inertia for k-means models with different values of 'k'
###############################################################################################

inertia = []
k_range = range(1, 10)

df_elbow = df_data.copy()
print(df_elbow.head())
# with scaling
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1200)
    pipeline = make_pipeline(robust_scaler, kmeans)
    pipeline.fit(df_elbow.values)
    labels = pipeline.predict(df_elbow)
    inertia.append(pipeline.named_steps['kmeans'].inertia_)

plt.figure(figsize=(15, 5))
plt.xlabel('k value', fontsize='x-large')
plt.ylabel('Model inertia', fontsize='x-large')
plt.title('Inertia(withScaledData)')
plt.plot(k_range, inertia, color='r')
plt.show()
fig.savefig('plots/clustering/Elbow_withScaledData.png')
# As we can see that the inertia value shows marginal decrement after k= 4,
# a k-means model with k=4(four clusters) is the most suitable for this task.
#########################################################################

# without scaling
print('Again')
print(df_elbow.head())
inertia = []
k_range = range(1, 10)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=1200)
    model.fit(df_elbow.values)
    inertia.append(model.inertia_)

# plotting the 'elbow curve'
plt.figure(figsize=(15, 5))
plt.xlabel('k value', fontsize='x-large')
plt.ylabel('Model inertia', fontsize='x-large')
plt.plot(k_range, inertia, color='r')
plt.title('Inertia(withoutScaledData)')
plt.show()
fig.savefig('plots/clustering/Elbow_withoutScaledData.png')

# As we can see that the inertia value shows marginal decrement after k= 2/3
# Optimal cluster can be 2 or 3
#########################################################################

# PCA : Principal Component Analysis

#########################################################################

# Pass Scaled data to PCA
# pca = PCA().fit(X_train_robust).transform(X_train_robust)
pca = PCA().fit(X_train_robust)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

print('Components :')
print(pca.components_)

# define the labels & title
plt.xlabel('Number of Components', fontsize=15)
plt.ylabel('Variance (%)', fontsize=15)
plt.title('Explained VarianceRatio (Cumulative)', fontsize=20)

# show the plot : Percentage of variance explained by each of the selected components.
plt.show()
fig.savefig('plots/clustering/PCA_ExplainedVarianceRatio.png')
#  we can see that we have 100% of variance explained with only two components.
#  This means that if we were to implement a PCA, we would select our number of components to be 2
#  If we won't get 100% explained the variance, but the general rule is to choose the minimum number of components
#  that demonstrates the highest amount of variance.


# Plot the explained variances : The amount of variance explained by each of the selected components.
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.title('Components Explained Variance')
plt.show()
fig.savefig('plots/clustering/PCA_ExplainedVariance.png')
#########################################################################
# PCA with 2 components
#########################################################################
# create a PCA modified data . We used scaled data
pca_data = PCA(n_components=2).fit(X_train_robust).transform(X_train_robust)

# store it in a new data frame
pca_data_df = pd.DataFrame(data=pca_data, columns=['principal component 1', 'principal component 2'])

# Assign 0th column of pca_features: xs
xs = pca_data[:, 0]
# Assign 1st column of pca_features: ys
ys = pca_data[:, 1]
# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)

# define a figure to plot Componenets
plt.figure()

# define the label and title
plt.xlabel('Principal Component 1', fontsize=15)
plt.ylabel('Principal Component 2', fontsize=15)
plt.title('2 Component PCA', fontsize=20)

# plot the figure
plt.scatter(pca_data_df['principal component 1'], pca_data_df['principal component 2'], c='royalBlue', s=50)
plt.show()
fig.savefig('plots/clustering/PCA_Component_Plot.png')

###################################################################################
# silhouette analysis to choose optimum value of k (clusters) for k Means : On Scaled Data

# Silhouette analysis can be used to study the separation distance between the resulting clusters.
# The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring
# clusters and thus provides a way to assess parameters like the number of clusters visually.
# This measure has a range of (-1, 1).
# Silhouette coefficients (as these values are referred to as) near +1 indicate
# that the sample is far away from the neighboring clusters. A value of 0 indicates
# that the sample is on or very close to the decision boundary between two neighboring clusters
# and negative values indicate that those samples might have been assigned to the wrong cluster.

# We will explore how the silhouette score changes within a limited range of clusters and will
# take max score to choose cluster
###############################################################################################


# define a dictionary that contains all information
results_dict = {}

# Define number of cluster
num_of_clusters = 10
k_range = range(2, num_of_clusters)
silhouette_coefficients =[]

for k in k_range:
    print("*" * 100)
    results_dict[k] = {}

    # create an instance of the model, and fit the training data(Scaled) to it.
    kmeans = KMeans(n_clusters=k, random_state=1200).fit(X_train_robust)

    # define silhouette score
    silhouette_score = metrics.silhouette_score(X_train_robust, kmeans.labels_, metric='euclidean')

    # store the metrics in result dictionary for a value of K
    results_dict[k]['silhouette_score'] = silhouette_score
    results_dict[k]['inertia'] = kmeans.inertia_
    results_dict[k]['score'] = kmeans.score
    results_dict[k]['model'] = kmeans

    # print the results
    print("Number of Clusters: {}".format(k))
    print('Silhouette Score:', silhouette_score)
    silhouette_coefficients.append(silhouette_score)

fig = plt.figure()
plt.style.use("fivethirtyeight")
plt.plot(k_range, silhouette_coefficients, color='g')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.title('Silhouette Score (with Robust Scaled Data')
plt.show()
fig.savefig('plots/clustering/SilhouetteScore_RobustScaled.png')
###################################################################################
# silhouette analysis to choose optimum value of k (clusters) for k Means : On PCA Dataset
###################################################################################

# define a dictionary
results_dict_pca = {}

# Define number of cluster
num_of_clusters = 10
k_range = range(2, num_of_clusters)
silhouette_coefficients = []
# run through each instance of K
for k in k_range:
    print("*" * 100)

    # define the next dictionary to hold all the results of this run.
    results_dict_pca[k] = {}

    # create an instance of the model, and fit the training data to it.
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pca_data_df)

    # define the silhouette score
    silhouette_score = metrics.silhouette_score(pca_data_df, kmeans.labels_, metric='euclidean')

    # store the different metrics
    results_dict_pca[k]['silhouette_score'] = silhouette_score
    results_dict_pca[k]['inertia'] = kmeans.inertia_
    results_dict_pca[k]['score'] = kmeans.score
    results_dict_pca[k]['model'] = kmeans

    # print the results
    print("Number of Clusters: {}".format(k))
    print('Silhouette Score:', silhouette_score)
    silhouette_coefficients.append(silhouette_score)

fig = plt.figure()
plt.style.use("fivethirtyeight")
plt.plot(k_range, silhouette_coefficients)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.title('Silhouette Score (with PCA DataSet')
plt.show()
fig.savefig('plots/clustering/SilhouetteScore_RobustScaled.png')

###################################################################################
# Evaluate Model
###################################################################################
# With Scaled Data
clusters = [2, 3]

for cluster in clusters:
    print('#' * 100)

    # define KMeans model for K
    kmeans = KMeans(n_clusters=cluster, random_state=1200)

    # pass the model through the visualizer
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')

    # fit the Scaled data
    visualizer.fit(X_train_robust)

    # show the chart
    visualizer.show()

# With PCA Data Set

clusters = [2, 3]

for cluster in clusters:

    print('#'*100)

    # define the model for K
    kmeans = KMeans(n_clusters=cluster, random_state=1200)

    # pass the model through the visualizer
    visualizer = SilhouetteVisualizer(kmeans)

    # fit the data
    visualizer.fit(pca_data)

    # show the chart
    visualizer.show()

# Graph with score 2 is a bad pick as few points are below average Silhoutte score.
# For n_clusters=3, all the plots are more or less of similar thickness and hence are of similar sizes ,
# all points are above average Silhoutte score , as can be considered as best ‘k’


clusters = [2, 3]

for cluster in clusters:
    print('-' * 100)

    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X_train_robust)

    # define the cluster centers
    cluster_centers = kmeans.cluster_centers_
    C1 = cluster_centers[:, 0]
    C2 = cluster_centers[:, 1]
    C3 = cluster_centers[:, 2]


    # create a new plot
    fig = plt.figure()
    ax = Axes3D(fig)

    # take the scaled data in this example.
    x = X_train_robust[:, 0]
    y = X_train_robust[:, 1]
    z = X_train_robust[:, 2]

    # define the axes labels
    column_names = df_data.columns
    ax.set_xlabel(column_names[0])
    ax.set_ylabel(column_names[1])
    ax.set_zlabel(column_names[2])

    # create a new plot
    ax.scatter(x, y, z, c=kmeans.labels_.astype(float), cmap='winter')
    ax.scatter(C1, C2, C3, marker="x", color='r')

    plt.title('Visualization of clustered data with {} clusters'.format(cluster), fontweight='bold')

    plt.show()
    fig.savefig('plots/clustering/KMeans_Scaled_'+str(cluster)+'_cluster_Scatter_plt.png')

clusters = [2, 3]

for cluster in clusters:
    print('-' * 100)

    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(pca_data)

    # define the cluster centers
    cluster_centers = kmeans.cluster_centers_
    C1 = cluster_centers[:, 0]
    C2 = cluster_centers[:, 1]

    # create a new plot
    plt.figure()

    # take the scaled data in this example.
    x = pca_data[:, 0]
    y = pca_data[:, 1]

    # define the axes labels
    column_names =['Component1', 'Component2']
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])

    # Visualize it:
    plt.scatter(x, y, c=kmeans.labels_.astype(float), cmap='winter')
    plt.scatter(C1, C2, marker="x", color='r')

    # Plot the clustered data
    plt.title('Visualization of clustered data with {} clusters'.format(cluster), fontweight='bold')
    plt.show()
    fig.savefig('plots/clustering/KMeans_PCA_' + str(cluster) + '_cluster_Scatter_plt.png')

#################################################################
# KMeans with 3 clusters
#################################################################
df_new = df_data.copy()

km_model = KMeans(n_clusters=3, random_state=1200)
pipeline = make_pipeline(robust_scaler, km_model)
pipeline.fit(df_new.values)
labels = pipeline.predict(df_new.values)
df_new['cluster'] = labels

# Plot

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig)

# axis
x = df_new['returnOnEquity'].values
y = df_new['returnOnAssets'].values
z = df_new['dividendYield'].values

colors = np.random.randint(224, size=(224))
# define the axes labels
column_names = df_new.columns
ax.set_xlabel(column_names[0] + '%')
ax.set_ylabel(column_names[1] + '%')
ax.set_zlabel(column_names[2] + '%')

# create a new plot
ax.scatter(x, y, z, c=df_new['cluster'],  cmap='winter')
ax.text2D(0.05, 0.95, "CLUSTERS from k-means algorithm with k = 3", transform=ax.transAxes)

centroids = pipeline.named_steps['kmeans'].cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
centroids_z = centroids[:, 2]
ax.scatter(centroids_x, centroids_y, centroids_z, marker='D', c='r')
plt.show()
fig.savefig('plots/clustering/Kmean_3_Plt.png')


final_cluster = df_new.merge(sector_df, how='left', on='symbol')
final_cluster.to_csv('../output_files/data_3_cluster.csv')
