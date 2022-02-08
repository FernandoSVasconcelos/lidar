import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cloudPoints

pd.options.mode.chained_assignment = None

def filtro(raw_points):
    new_df = []
    media_z = raw_points['Z'].mean()
    desvio_z = raw_points['Z'].std()
    for index, row in raw_points.iterrows():
         if (row['reflectivity'] > 101) and (row['Y'] > 0) and (row['Z'] > media_z + desvio_z) and abs(row['X'] < 30):   
            new_df.append(row)
                    
    new_df = pd.DataFrame(new_df)
    return new_df

def kmeans(new_data, N_CLUSTERS):
    X = new_data.iloc[:, 0:3].values
    data_list = []
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple', 'brown']

    k_means_optimum = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++',  random_state=42)
    y = k_means_optimum.fit_predict(X)

    new_data['cluster'] = y 

    for i in range(N_CLUSTERS):
        data_list.append(new_data[new_data.cluster == i])

    raw_points = new_data
    filtered_df = filtro(raw_points)
    points = cloudPoints.slam([raw_points], [[0,0], [0,0]], [[0,0,0], [0,0,0]])
    cloudPoints.generate_mesh(points)

    '''kplot = plt.axes(projection='3d')

    for i in range(N_CLUSTERS):
        kplot.scatter3D(data_list[i]['X'], data_list[i]['Y'], data_list[i]['Z'], c=f'{colors[i]}', label = f'Cluster {i}')
    

    kplot.set_xlabel('$X$', fontsize=20, rotation=0)
    kplot.set_ylabel('$Y$', fontsize=20, rotation=0)
    kplot.set_zlabel('$Z$', fontsize=20, rotation=0)
    
    plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 101)
    plt.legend()
    plt.title("Kmeans")
    plt.show()'''
    return new_data

def getDistancia(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        x = new_data['X'].loc[new_data['X'].abs().idxmin()]
        y = new_data['Y'].loc[new_data['Y'].abs().idxmin()]
        z = new_data['Z'].loc[new_data['Z'].abs().idxmin()]

        root = x**2 + y**2 + z**2
        distancia = math.sqrt(root)

        return (distancia)
    except Exception as e:
        print(f"Sem dados: {e}")

def getAltura(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        my_max = new_data['Z'].loc[new_data['Z'].idxmax()]
        my_min = new_data['Z'].loc[new_data['Z'].idxmin()]
        media = new_data['Z'].mean()
        desvio = new_data['Z'].std()
        variancia = math.sqrt(desvio)
        altura = my_max - media
        if(desvio > 1):
            altura = altura / variancia

        return altura
    except Exception as e:
        print(f"Sem dados: {e}")

def getClusterSize(data, N_CLUSTERS):
    list_alt = []
    try:
        for i in range(N_CLUSTERS):
            new_data = data[data.cluster == i]
            my_max = new_data['Z'].loc[new_data['Z'].idxmax()]
            my_min = new_data['Z'].loc[new_data['Z'].idxmin()]
            if(my_min < 0):
                altura = abs(my_max) + abs(my_min)
            else:
                altura = abs(my_max) - abs(my_min)
            list_alt.append(altura)
            print(f"Tamanho do cluster {i}: {list_alt[i]:.2f}")
        max_value = max(list_alt)
        max_index = list_alt.index(max_value)
        print('---------------------------------------------------------')
        print(f"Cluster Selecionado: {max_index}")
        print('---------------------------------------------------------')

        return max_index
    except Exception as e:
        print(f"Sem dados: {e}")

def getQuadrante(data, side):
    try:
        if(side == 'top-right'):
            new_data = (data[data['X'] > 0])
        elif(side == 'top-left'):
            new_data = (data[data['X'] < 0])
        elif(side == 'bottom-right'):
            new_data = (data[data['X'] > 0])
        elif(side == 'bottom-left'):
            new_data = (data[data['X'] < 0])
        elif(side == 'Null'):
            new_data = data
        return new_data
    except Exception as e:
        print(f"Sem dados: {e}")

def wire_compare(processed_data):
    res = []
    new_df = processed_data
    drop_index = []
    media_z = new_df['Z'].mean()
    desvio_z = new_df['Z'].std()
    for index, row in processed_data.iterrows():
        if (row['reflectivity'] > 101) and (row['Y'] > 0) and (row['Z'] > media_z + desvio_z) and abs(row['X'] < 30):
            res.append(row)
            drop_index.append(index)
    print(f"Há {len(res)} possíveis pontos do fio no quadrante da árvore.")
    print('---------------------------------------------------------')
    for index in drop_index:
        new_df = new_df.drop(index)
    
    return len(res), new_df

def dist_compare(filtered_data, processed_data, max_index):
    new_data = processed_data[processed_data.cluster == max_index]
    distX = []
    distY = []
    distZ = []
    distY = []
    distT = []
    x = []
    y = []
    z = []

    x1 = new_data['X']
    y1 = new_data['Y']
    z1 = new_data['Z']
    
    for index, row in filtered_data.iterrows():
        distX.append(abs(x1 - row.X))
        distY.append(abs(y1 - row.Y))
        distZ.append(abs(z1 - row.Z))

    for i in range(len(distX)):
        for index, values in distX[i].iteritems():
            x.append(values)
        for index, values in distY[i].iteritems():
            y.append(values)
        for index, values in distZ[i].iteritems():
            z.append(values)

    for i in range(len(x)):
        root = x[i]**2 + y[i]**2 + z[i]**2
        distT.append(math.sqrt(root))

    if(min(distT) == 0):
        print(f"O fio está em contato com a árvore!")
    else:
        print(f"A distância mínima entre o fio e a árvore é de {min(distT):.3f} metros.")
    print('---------------------------------------------------------')
    return min(distT)

def deleta_discrepantes(data, N_CLUSTERS):
    data_clusters = []
    new_data = pd.DataFrame()
    for k in range(N_CLUSTERS):
        data_clusters.append(data[data.cluster == k])
        data_clusters[k] = data_clusters[k].sort_values('Z')
        
        desvio_X = data_clusters[k].X.std()
        desvio_Y = data_clusters[k].Y.std()
        desvio_Z = data_clusters[k].Z.std()
        
        x = []
        x_remove = []
        y = []
        y_remove = []
        dif_x = 0
        for index, row in data_clusters[k].iterrows():
            x.append(row['X'])
            y.append(row['Y'])
        len_x = len(x)
        i = 0
        while i < len_x:
            try:
                if i > 0:
                    dif_x = abs(x[i] - x[i - 1])
                    if dif_x > desvio_X:
                        x_remove.append(x[i])
                        x.pop(i)
                    else:
                        i += 1
                else:
                    i += 1
                        
                len_x = len(x)
            except:
                break
        
        for index, row in data_clusters[k].iterrows():
            for line in x_remove:
                if line == row.X:
                    try:
                        data_clusters[k] = data_clusters[k].drop(index)
                    except:
                        pass

        new_data = pd.concat(data_clusters)

    return new_data

def deleta_verticais(data, N_CLUSTERS):
    data_clusters = []
    new_data = pd.DataFrame()
    for k in range(N_CLUSTERS):
        try:
            data_clusters.append(data[data.cluster == k])
            data_clusters[k] = data_clusters[k].sort_values('Z')
            xb = data_clusters[k]['X'].iloc[-1]
            xa = data_clusters[k]['X'].iloc[0]
            yb = data_clusters[k]['Y'].iloc[-1]
            ya = data_clusters[k]['Y'].iloc[0]
            zb = data_clusters[k]['Z'].iloc[-1]
            za = data_clusters[k]['Z'].iloc[0]
            d1 = (xb - xa)**2 + (yb - ya)**2
            d2 = (zb - za)**2
            if(d2 > d1):
                data_clusters[k] = pd.DataFrame()
        except:
            pass

    new_data = pd.concat(data_clusters)
    return new_data

def get_Filter(raw_points):
    new_df = []
    media_z = raw_points.Z.mean()
    desvio_z = raw_points.Z.std()
    for index, row in raw_points.iterrows():
         if (row['X'] < 5) and (row['Z'] > media_z + desvio_z):   
            new_df.append(row)
                    
    new_df = pd.DataFrame(new_df)
    print(new_df.Z.mean())
    print(new_df.reflectivity.mean())
    return new_df

def main(path, quadrante):
    data = pd.read_csv(path)
    #----------------------Filtro---------------------------
    new_data = data[['X', 'Y', 'Z', 'reflectivity']].copy()
    #########################################################
    #-------------------1º Kmeans----------------------------
    processed_data = kmeans(new_data, 5)
    processed_data = getQuadrante(new_data, quadrante)
    filtered_data = filtro(processed_data)
    #########################################################
    #teste_df = get_Filter(new_data)
    #teste_df = kmeans(teste_df, 5)
    #-------------------2º Kmeans----------------------------
    res, processed_data = wire_compare(processed_data)
    try:
        processed_data = kmeans(processed_data, 8)
    except Exception as e:
        print(f"Sem dados processados: {e}")
    #########################################################
    #--------------------3º Kmeans---------------------------
    try:
        if(filtered_data.shape[0] > 0):
            filtered_data = deleta_discrepantes(filtered_data, N_CLUSTERS = 8)
            filtered_data = deleta_verticais(filtered_data, N_CLUSTERS = 8)
            if(len(filtered_data) < 4):
                print(f"Sem fios no quadrante da árvore!")
                print('---------------------------------------------------------')
            else:
                filtered_data = kmeans(filtered_data, 4)
        else:
            print(f"Sem dados filtrados!")
    except Exception as e:
        print(f"Sem dados filtrados: {e}")
    #########################################################
    #-------------------------Cálculos-----------------------
    max_index = getClusterSize(processed_data, N_CLUSTERS = 8)
    try:
        distancia = getDistancia(processed_data, max_index)
        altura = getAltura(processed_data, max_index)
        try:
            dist_compare(filtered_data, processed_data, max_index)
        except:
            print('Sem fios detectados na captura!')
            print('---------------------------------------------------------')
    except Exception as e:
        print(f"Sem dados no quadrante: {e}")
        return None, None, None
    #########################################################
    return distancia, altura, processed_data

if __name__ == '__main__':
    path = "new_csv/20211210121705.lidar.csv"
    quadrante = 'top-left'

    distancia, altura, _ = main(path, quadrante)
    if(distancia is None):
        print("Sem pontos no quadrante!")
    else:
        print(f"Altura: {altura:.2f} metros")
        print(f"Distancia: {distancia:.2f} metros")