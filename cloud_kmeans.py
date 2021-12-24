import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def filtro(raw_points):
	new_df = []
	for index, row in raw_points.iterrows():
		if (row['intensity'] < 10) and (row['Points_m_XYZ:2'] > 5):
			if (abs(row['Points_m_XYZ:0']) > 5) and (abs(row['Points_m_XYZ:0']) < 20):
				if (abs(row['Points_m_XYZ:1']) > 5) and (abs(row['Points_m_XYZ:1']) < 50):
					new_df.append(row)
	new_df = pd.DataFrame(new_df)
	return new_df

def find_clusters(X):
    wcss = []
    for i in range(1,11):
        k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
        k_means.fit(X)
        wcss.append(k_means.inertia_)

    plt.plot(np.arange(1,11),wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.show()

def kmeans(new_data):
    X = new_data.iloc[:,0:3].values

    #find_clusters(X)

    k_means_optimum = KMeans(n_clusters = 4, init = 'k-means++',  random_state=42)
    y = k_means_optimum.fit_predict(X)

    new_data['cluster'] = y 

    data1 = new_data[new_data.cluster==0]
    data2 = new_data[new_data.cluster==1]
    data3 = new_data[new_data.cluster==2]
    data4 = new_data[new_data.cluster==3]

    kplot = plt.axes(projection='3d')
    
    kplot.scatter3D(data1['Points_m_XYZ:0'], data1['Points_m_XYZ:1'], data1['Points_m_XYZ:2'], c='red', label = 'Cluster 1')
    kplot.scatter3D(data2['Points_m_XYZ:0'], data2['Points_m_XYZ:1'], data2['Points_m_XYZ:2'],c ='green', label = 'Cluster 2')
    kplot.scatter3D(data3['Points_m_XYZ:0'], data3['Points_m_XYZ:1'], data3['Points_m_XYZ:2'],c ='blue', label = 'Cluster 3')
    kplot.scatter3D(data4['Points_m_XYZ:0'], data4['Points_m_XYZ:1'], data4['Points_m_XYZ:2'],c ='yellow', label = 'Cluster 4')
    plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
    plt.legend()
    plt.title("Kmeans")
    plt.show()
    return new_data

def getDistancia(data, max_index):
    try:
        new_data = data[data.cluster == max_index]

        desvio_X = new_data['Points_m_XYZ:0'].std()
        desvio_Y = new_data['Points_m_XYZ:1'].std()
        desvio_Z = new_data['Points_m_XYZ:2'].std()

        media_X = new_data['Points_m_XYZ:0'].mean()
        media_Y = new_data['Points_m_XYZ:1'].mean()
        media_Z = new_data['Points_m_XYZ:2'].mean()

        distancia = abs(media_X) + abs(media_Y) + abs(media_Z)
        print(f"Distância: {distancia:.2f} metros")

        print('---------------------------------------------------------')
        return distancia
    except Exception as e:
        print(f"Sem dados: {e}")

def getAltura(data, max_index):
    try:
        new_data = data[data.cluster == max_index]

        my_max = new_data['Points_m_XYZ:2'].loc[new_data['Points_m_XYZ:2'].idxmax()]
        my_min = new_data['Points_m_XYZ:2'].loc[new_data['Points_m_XYZ:2'].idxmin()]

        if(my_min < 0):
            altura = abs(my_max) + abs(my_min)
        else:
            altura = abs(my_max) - abs(my_min)


        print(f"Altura: {altura:.2f} metros")
        print('---------------------------------------------------------')
        return altura
    except Exception as e:
        print(f"Sem dados: {e}")

def getClusterSize(data):
    try:
        cluster = []

        for i in range(4):
            cluster.append(data[data.cluster == i].shape[0])
            print(f"Tamanho do cluster {i}: {cluster[i]}")

        max_value = max(cluster)
        max_index = cluster.index(max_value)
        print('---------------------------------------------------------')
        return max_index
    except Exception as e:
        print(f"Sem dados: {e}")

def getQuadrante(data, side):
    try:
        if(side == 'top-rigth'):
            aux_data = (data[data['Points_m_XYZ:0'] > 0])
            new_data = (aux_data[aux_data['Points_m_XYZ:1'] > 0])
        elif(side == 'top-left'):
            aux_data = (data[data['Points_m_XYZ:0'] < 0])
            new_data = (aux_data[aux_data['Points_m_XYZ:1'] > 0])
        elif(side == 'bottom-rigth'):
            aux_data = (data[data['Points_m_XYZ:0'] > 0])
            new_data = (aux_data[aux_data['Points_m_XYZ:1'] < 0])
        elif(side == 'bottom-left'):
            aux_data = (data[data['Points_m_XYZ:0'] < 0])
            new_data = (aux_data[aux_data['Points_m_XYZ:1'] < 0])
        return new_data
    except Exception as e:
        print(f"Sem dados: {e}")

def main(path, quadrante):
    data = pd.read_csv(path)
    #data = filtro(data)
    new_data = data[['Points_m_XYZ:0', 'Points_m_XYZ:1', 'Points_m_XYZ:2']].copy()

    processed_data = kmeans(new_data)
    processed_data = getQuadrante(processed_data, quadrante)

    max_index = getClusterSize(processed_data)
    distancia = getDistancia(processed_data, max_index)
    altura = getAltura(processed_data, max_index)

    try:
        processed_data = kmeans(processed_data)
    except Exception as e:
        print(f"Sem dados: {e}")
    
    return distancia, altura, processed_data

if __name__ == '__main__':
    path = "new_csv/cap21.csv"
    main(path, 'top-left')