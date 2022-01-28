import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N_CLUSTERS = 5

def filtro(raw_points):
    new_df = []
    for index, row in raw_points.iterrows():
        if (row['reflectivity'] < 10):
            if ((abs(row['X']) > 1) and (abs(row['Y']) < 7)) or ((abs(row['X']) > 3) and (abs(row['Y']) > 7)):
                new_df.append(row)
                    
    new_df = pd.DataFrame(new_df)
    return new_df

def find_clusters(X):
    wcss = []
    for i in range(1,11):
        k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
        k_means.fit(X)
        wcss.append(k_means.inertia_)

    plt.plot(np.arange(1, 11), wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.show()

def kmeans(new_data):
    X = new_data.iloc[:, 0:3].values

    #find_clusters(X)

    k_means_optimum = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++',  random_state=42)
    y = k_means_optimum.fit_predict(X)

    new_data['cluster'] = y 

    data1 = new_data[new_data.cluster == 0]
    data2 = new_data[new_data.cluster == 1]
    data3 = new_data[new_data.cluster == 2]
    data4 = new_data[new_data.cluster == 3]
    data5 = new_data[new_data.cluster == 4]

    kplot = plt.axes(projection='3d')
    
    kplot.scatter3D(data1['X'], data1['Y'], data1['Z'], c='red', label = 'Cluster 1')
    kplot.scatter3D(data2['X'], data2['Y'], data2['Z'],c ='green', label = 'Cluster 2')
    kplot.scatter3D(data3['X'], data3['Y'], data3['Z'],c ='blue', label = 'Cluster 3')
    kplot.scatter3D(data4['X'], data4['Y'], data4['Z'],c ='yellow', label = 'Cluster 4')
    kplot.scatter3D(data5['X'], data5['Y'], data5['Z'],c ='black', label = 'Cluster 5')
    plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
    plt.legend()
    plt.title("Kmeans")
    plt.show()
    return new_data

def getDistancia(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        desvio_X = new_data['X'].std()
        desvio_Y = new_data['Y'].std()
        desvio_Z = new_data['Z'].std()

        media_X = new_data['X'].mean()
        media_Y = new_data['Y'].mean()
        media_Z = new_data['Z'].mean()

        distancia = abs(media_X) + abs(media_Y) + abs(media_Z)

        return distancia
    except Exception as e:
        print(f"Sem dados: {e}")

def getAltura(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        my_max = new_data['Z'].loc[new_data['Z'].idxmax()]
        my_min = new_data['Z'].loc[new_data['Z'].idxmin()]

        if(my_min < 0):
            altura = abs(my_max) + abs(my_min)
        else:
            altura = abs(my_max) - abs(my_min)

        return altura
    except Exception as e:
        print(f"Sem dados: {e}")

def getClusterSize(data):
    try:
        cluster = []
        for i in range(N_CLUSTERS):
            cluster.append(data[data.cluster == i].shape[0])
            print(f"Tamanho do cluster {i}: {cluster[i]}")
        print('---------------------------------------------------------')
        max_value = max(cluster)
        max_index = cluster.index(max_value)
        return max_index
    except Exception as e:
        print(f"Sem dados: {e}")

def getQuadrante(data, side):
    try:
        if(side == 'top-right'):
            aux_data = (data[data['X'] > 0])
            new_data = (aux_data[aux_data['Y'] > 0])
        elif(side == 'top-left'):
            aux_data = (data[data['X'] < 0])
            new_data = (aux_data[aux_data['Y'] > 0])
        elif(side == 'bottom-right'):
            aux_data = (data[data['X'] > 0])
            new_data = (aux_data[aux_data['Y'] < 0])
        elif(side == 'bottom-left'):
            aux_data = (data[data['X'] < 0])
            new_data = (aux_data[aux_data['Y'] < 0])
        elif(side == 'Null'):
            new_data = data
        return new_data
    except Exception as e:
        print(f"Sem dados: {e}")

def wire_compare(filtered_data, processed_data):
    res = []
    processed_data = processed_data[['X', 'Y', 'Z']].copy()
    
    x1 = processed_data['X']
    y1 = processed_data['Y']
    z1 = processed_data['Z']
    for index, row in filtered_data.iterrows():
        x2 = row['X']
        y2 = row['Y']
        z2 = row['Z']
        if(((x1 == x2) & (y1 == y2) & (z1 == z2)).any()):
            res.append(row)
    print(f"Há {len(res)} possíveis pontos do fio no quadrante da árvore.")
    print('---------------------------------------------------------')
    return len(res)

def dist_compare(filtered_data, processed_data):
    distX = []
    distY = []
    distZ = []
    minX = []
    minY = []
    minZ = []
    x1 = processed_data['X']
    y1 = processed_data['Y']
    z1 = processed_data['Z']
    for index, row in filtered_data.iterrows():
        distX.append(abs(x1 - row.X))
        distY.append(abs(y1 - row.Y))
        distZ.append(abs(z1 - row.Z))
    
    for i in range (len(distX)):
        minX.append(min(distX[i]))
        minY.append(min(distY[i]))
        minZ.append(min(distZ[i]))

    print(f"Distância mínima no eixo X: {min(minX)}")
    print(f"Distância mínima no eixo Y: {min(minY)}")
    print(f"Distância mínima no eixo Z: {min(minZ)}")
    soma = [min(minX), min(minY), min(minZ)]
    print('---------------------------------------------------------')
    if(min(soma) == 0):
        print(f"O fio está em contato com a árvore!")
    else:
        print(f"A distância mínima entre o fio e a árvore é de {min(soma):.3f} metros.")
    print('---------------------------------------------------------')

def main(path, quadrante):
    data = pd.read_csv(path)

    filtered_data = filtro(data)
    filtered_data = filtered_data[['X', 'Y', 'Z']].copy()

    new_data = data[['X', 'Y', 'Z']].copy()

    processed_data = kmeans(new_data)
    processed_data = getQuadrante(new_data, quadrante)
    try:
        processed_data = kmeans(processed_data)
    except Exception as e:
        print(f"Sem dados processados: {e}")
    
    try:
        filtered_data = kmeans(filtered_data)
    except Exception as e:
        print(f"Sem dados filtrados: {e}")

    res = wire_compare(filtered_data, processed_data)

    max_index = getClusterSize(processed_data)
    try:
        distancia = getDistancia(processed_data, max_index)
        altura = getAltura(processed_data, max_index)
        dist_compare(filtered_data, processed_data)
    except Exception as e:
        print(f"Sem dados no quadrante: {e}")
        return None, None, None

    return distancia, altura, processed_data
if __name__ == '__main__':
    path = "new_csv/20211210121705.lidar.csv"
    quadrante = 'top-right'

    distancia, altura, _ = main(path, quadrante)
    if(distancia is None):
        print("Sem pontos no quadrante!")
    else:
        print(f"Altura: {altura:.2f} metros")
        print(f"Distancia: {distancia:.2f} metros")