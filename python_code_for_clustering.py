import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from fbprophet import Prophet

plt.style.use("ggplot")

df = pd.read_csv('prediction_training1.csv')

print(df.tail()._get_values)

m = Prophet()
m.fit(df);

future = m.make_future_dataframe(periods=5,freq='min')
print(future.tail())

forecast = m.predict(future)
forecasted_ram_usage=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecasted_ram_usage)

df = pd.read_csv('prediction_training2.csv')

print(df.tail())

m = Prophet()
m.fit(df);

future = m.make_future_dataframe(periods=5,freq='min')
print(future.tail())

forecast = m.predict(future)
forecasted_disk_throughput=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(forecasted_disk_throughput)

print(" ")

print("Busy Server Thread: 20. CPU Usage (%): 10")

with open('Training_Metrics_Data.csv') as csvfile:
    colormapArr = ['Red', 'Blue', 'Green', 'black', 'orange']
    readCSV = csv.reader(csvfile)
    labels = []
    rows = []
    for idx, row in enumerate(readCSV, 0):
        if idx == 0:
            labels.append(row[2:4])
        else:
            rows.append(row[2:4])

    x = pd.DataFrame(rows, columns=labels[0])
    df = x.get('BusyServerThreads')
    sf = x.get('CPUUsageinPer')

    kmeans = KMeans(n_clusters=5).fit(x)
    print(" ")
    print("Number of Clusters: " + str(kmeans.n_clusters))
    print(" ")
    print("Centroids: ")
    print(kmeans.cluster_centers_)
    print(" ")

    highestClusterVal = kmeans.predict([[20, 10]])
    print("WebI: Training Data --> Predicted value falls in Cluster: " + colormapArr[highestClusterVal[0]] + " i.e., Centroid: "
          + str(kmeans.cluster_centers_[highestClusterVal[0]]))
    print(" ")

    colormap = np.array(colormapArr)
    # plot = plt.scatter(df, sf, c=colormap[kmeans.labels_])
    plt.figure(1).suptitle("WebI Server: Training Data - Cluster")
    plt.scatter(df, sf, c=colormap[kmeans.labels_])
    plt.xlabel("Busy Server Threads")
    plt.ylabel("CPU Usage %")
    # plt.show()

    testLabels = []
    with open('Test_Metrics_Data.csv') as csvfile:
        readCSV = csv.reader(csvfile)
        for idx, row in enumerate(readCSV, 0):
            if idx == 0:
                testLabels.append(row[0:])
            else:
                rows.append(row)
    x = pd.DataFrame(rows, columns=testLabels[0])

    df = x.get('BusyServerThreads')
    sf = x.get('CPUUsageinPer')

    kmeans = KMeans(n_clusters=5).fit(x)
    print(kmeans.cluster_centers_)
    colormap = np.array(colormapArr)
    # plot = plt.scatter(df, sf, c=colormap[kmeans.labels_])

    highestClusterVal = kmeans.predict([[20, 10]])
    print(" ")
    print("WebI: Test Data --> Predicted value falls in Cluster: " + colormapArr[highestClusterVal[0]] + " i.e., Centroid: "
          + str(kmeans.cluster_centers_[highestClusterVal[0]]))

    plt.figure(2).suptitle("WebI Server: Test Data - Cluster")
    plt.xlabel("Busy Server Threads")
    plt.ylabel("CPU Usage %")
    plt.scatter(df, sf, c=colormap[kmeans.labels_])
    plt.show()
