import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cloudPoints

pd.options.mode.chained_assignment = None

def filtro(raw_points):
    new_df = []
    corte_z = (abs(raw_points['Z'].mean())) + (math.sqrt(abs(raw_points['Z'].std())))
    corte_x = (abs(raw_points['X'].mean())) + (abs(raw_points['X'].std()))

    print('---------------------------------------------------------')
    print(f"Limiar de corte em Z: {corte_z:.2f} metros")
    print(f"Limiar de corte em X: {corte_x:.2f} metros")
    print('---------------------------------------------------------')

    for _, row in raw_points.iterrows():
         if (row['reflectivity'] > 101) and (row['Z'] > corte_z) and abs(row['X'] < corte_x):   
            new_df.append(row)         
    new_df = pd.DataFrame(new_df)
    return new_df

def plot(data):
    points = cloudPoints.slam([data], [[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0]])
    cloudPoints.generate_mesh(points)

def kmeans(new_data, N_CLUSTERS):
    X = new_data.iloc[:, 0:3].values
    data_list = []
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple', 'brown']

    k_means_optimum = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++',  random_state = 42)
    y = k_means_optimum.fit_predict(X)
    new_data['cluster'] = y 

    for i in range(N_CLUSTERS):
        data_list.append(new_data[new_data.cluster == i])
    raw_points = new_data
    
    '''kplot = plt.axes(projection='3d')
    for i in range(N_CLUSTERS):
        kplot.scatter3D(data_list[i]['X'], data_list[i]['Y'], data_list[i]['Z'], c=f'{colors[i]}', label = f'Cluster {i}')
    kplot.set_xlabel('$X$', fontsize=20, rotation=0)
    kplot.set_ylabel('$Y$', fontsize=20, rotation=0)
    kplot.set_zlabel('$Z$', fontsize=20, rotation=0)
    
    plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 130)
    plt.legend()
    plt.title("Kmeans")
    plt.show()'''
    return new_data

def getDistancia(processed_data, max_index):
    distX = []
    distY = []
    distZ = []
    distT = []
    new_data = processed_data[processed_data.cluster == max_index]
    try:
        for _, row in new_data.iterrows():
            distX.append(row.X)
            distY.append(row.Y)
            distZ.append(row.Z)
        for i in range(len(distX)):
            root = distX[i]**2 + distY[i]**2 + distZ[i]**2
            distT.append(math.sqrt(root))
        return (min(distT))
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def getAltura(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        my_max = new_data['Z'].loc[new_data['Z'].idxmax()]
        media = new_data['Z'].mean()
        desvio = new_data['Z'].std()
        variancia = math.sqrt(desvio)
        altura = my_max - media

        if(desvio > 1):
            altura = altura / variancia
        return altura
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def getClusterSize(data, N_CLUSTERS):
    list_stdX = []
    list_stdY = []
    try:
        for i in range(N_CLUSTERS):
            new_data = data[data.cluster == i]
            list_stdX.append(new_data['X'].std())
            list_stdY.append(new_data['Y'].std())
        max_valueX = np.argsort(list_stdX)[::-1][:N_CLUSTERS]
        max_valueY = np.argsort(list_stdY)[::-1][:N_CLUSTERS]
        print(f"Maiores Clusters por desvio em X: {max_valueX}")
        print(f"Maiores Clusters por desvio em Y: {max_valueY}")
        print('---------------------------------------------------------')
        return max_valueX
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

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
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def wire_compare(processed_data):
    res = []
    drop_index = []
    new_df = processed_data
    corte_z = (abs(new_df['Z'].mean())) + (math.sqrt(abs(new_df['Z'].std())))
    corte_x = (abs(new_df['X'].mean())) + (abs(new_df['X'].std()))

    for index, row in processed_data.iterrows():
        if (row['reflectivity'] > 101) and (row['Z'] > corte_z) and abs(row['X'] < corte_x):
            res.append(row)
            drop_index.append(index)
    print(f"Há {len(res)} possíveis pontos de um possível fio no quadrante da árvore.")
    print('---------------------------------------------------------')
    for index in drop_index:
        new_df = new_df.drop(index)
    return  new_df

def dist_compare(filtered_data, processed_data, max_index):
    distX = []
    distY = []
    distZ = []
    distT = []
    x = []
    y = []
    z = []

    new_data = processed_data[processed_data.cluster == max_index]
    x1 = new_data['X']
    y1 = new_data['Y']
    z1 = new_data['Z']

    for _, row in filtered_data.iterrows():
        distX.append(abs(x1 - row.X))
        distY.append(abs(y1 - row.Y))
        distZ.append(abs(z1 - row.Z))
    for i in range(len(distX)):
        for _, values in distX[i].iteritems():
            x.append(values)
        for _, values in distY[i].iteritems():
            y.append(values)
        for _, values in distZ[i].iteritems():
            z.append(values)
    for i in range(len(x)):
        root = x[i]**2 + y[i]**2 + z[i]**2
        distT.append(math.sqrt(root))
    if(min(distT) == 0):
        print(f"O fio está em contato com a árvore!")
    else:
        if(min(distT) < 1):
            print(f"A distância mínima entre um possível contato entre fio e árvore é de {min(distT)*100:.2f} centímetros.")
        else:
            print(f"A distância mínima entre um possível contato entre fio e árvore é de {min(distT):.2f} metros.")
    print('---------------------------------------------------------')
    return min(distT)

def deleta_discrepantes(data, N_CLUSTERS):
    data_clusters = []
    new_data = pd.DataFrame()

    for k in range(N_CLUSTERS):
        x = []
        x_remove = []
        dif_x = 0

        data_clusters.append(data[data.cluster == k])
        data_clusters[k] = data_clusters[k].sort_values('Z')
        desvio_X = data_clusters[k].X.std()

        for index, row in data_clusters[k].iterrows():
            x.append(row['X'])
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
            continue
    new_data = pd.concat(data_clusters)
    return new_data

def soma10(df, filtered_data):
    filtered_data['Z'] = filtered_data['Z'] + 15
    new_df = pd.concat([df, filtered_data])
    new_df = plot(new_df)

def find_tree(data, N_CLUSTERS):
    list_alt = []
    list_index = []
    try:
        for i in range(N_CLUSTERS):
            new_data = data[data.cluster == i]
            z0 = new_data['Z'].loc[new_data['Z'].idxmin()]
            z25 = new_data['Z'].quantile(q = 0.25)
            z50 = new_data['Z'].mean()
            z75 = new_data['Z'].quantile(q = 0.75)
            z100 = new_data['Z'].loc[new_data['Z'].idxmax()]
            #-------------------------------------------------------------------
            x0 = data[data['Z'] == z0]
            if x0.shape[0] > 0:
                x0 = x0['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z0:
                        list_alt.append(row.X)
                x0 = min(list_alt)
                list_alt = []

            x25 = data[data['Z'] == z25]
            if x25.shape[0] > 0:
                x25 = x25['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z25:
                        list_alt.append(row.X)
                x25 = min(list_alt)
                list_alt = []

            x50 = data[data['Z'] == z50]
            if x50.shape[0] > 0:
                x50 = x50['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z50:
                        list_alt.append(row.X)
                x50 = min(list_alt)
                list_alt = []

            x75 = data[data['Z'] == z75]
            if x75.shape[0] > 0:
                x75 = x75['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z75:
                        list_alt.append(row.X)
                x75 = min(list_alt)
                list_alt = []

            x100 = data[data['Z'] == z100]
            if x100.shape[0] > 0:
                x100 = x100['X'].mean()
            else:
                for index, row in x0.iterrows():
                    if row.Z < z100:
                        list_alt.append(row.X)
                x100 = max(list_alt)
                list_alt = []
            #--------------------------------IF----------------------------------
            if(x0 > x25) and (x25 >= x50) and (x50 <= x75) and (x75 < x100):
                list_index.append(i)
                continue
            #--------------------------------X0----------------------------------
            x0 = data[data['Z'] == z0]
            if x0.shape[0] > 0:
                x0 = x0['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z0:
                        list_alt.append(row.X)
                x0 = min(list_alt)
                list_alt = []
            #--------------------------------X25---------------------------------
            x25 = data[data['Z'] == z25]
            if x25.shape[0] > 0:
                x25 = x25['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z25:
                        list_alt.append(row.X)
                x25 = min(list_alt)
                list_alt = []
            #--------------------------------X50---------------------------------
            x50 = data[data['Z'] == z50]
            if x50.shape[0] > 0:
                x50 = x50['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z50:
                        list_alt.append(row.X)
                x50 = min(list_alt)
                list_alt = []
            #--------------------------------X75---------------------------------
            x75 = data[data['Z'] == z75]
            if x75.shape[0] > 0:
                x75 = x75['X'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z75:
                        list_alt.append(row.X)
                x75 = min(list_alt)
                list_alt = []
            #--------------------------------X100--------------------------------
            x100 = data[data['Z'] == z100]
            if x100.shape[0] > 0:
                x100 = x100['X'].mean()
            else:
                for index, row in x0.iterrows():
                    if row.Z < z100:
                        list_alt.append(row.X)
                x100 = max(list_alt)
                list_alt = []
            #--------------------------------IF----------------------------------
            if(x0 < x25) and (x25 <= x50) and (x50 >= x75) and (x75 > x100):
                list_index.append(i)
                continue
            #--------------------------------Y0----------------------------------
            y0 = data[data['Z'] == z0]
            if y0.shape[0] > 0:
                y0 = y0['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z0:
                        list_alt.append(row.Y)
                y0 = min(list_alt)
                list_alt = []
            #--------------------------------Y25---------------------------------
            y25 = data[data['Z'] == z25]
            if y25.shape[0] > 0:
                y25 = y25['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z25:
                        list_alt.append(row.Y)
                y25 = min(list_alt)
                list_alt = []
            #--------------------------------Y50---------------------------------
            y50 = data[data['Z'] == z50]
            if y50.shape[0] > 0:
                y50 = y50['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z50:
                        list_alt.append(row.Y)
                y50 = min(list_alt)
                list_alt = []
            #--------------------------------Y75---------------------------------
            y75 = data[data['Z'] == z75]
            if y75.shape[0] > 0:
                y75 = y75['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z75:
                        list_alt.append(row.Y)
                y75 = min(list_alt)
                list_alt = []
            #--------------------------------Y100--------------------------------
            y100 = data[data['Z'] == z100]
            if y100.shape[0] > 0:
                y100 = y100['Y'].mean()
            else:
                for index, row in x0.iterrows():
                    if row.Z < z100:
                        list_alt.append(row.Y)
                y100 = max(list_alt)
                list_alt = []
            #--------------------------------IF----------------------------------
            if(y0 > y25) and (y25 >= y50) and (y50 <= y75) and (y75 < y100):
                list_index.append(i)
                continue
            #--------------------------------Y0----------------------------------
            y0 = data[data['Z'] == z0]
            if y0.shape[0] > 0:
                y0 = y0['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z0:
                        list_alt.append(row.Y)
                y0 = min(list_alt)
                list_alt = []
            #--------------------------------Y25---------------------------------
            y25 = data[data['Z'] == z25]
            if y25.shape[0] > 0:
                y25 = y25['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z25:
                        list_alt.append(row.Y)
                y25 = min(list_alt)
                list_alt = []
            #--------------------------------Y50---------------------------------
            y50 = data[data['Z'] == z50]
            if y50.shape[0] > 0:
                y50 = y50['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z50:
                        list_alt.append(row.Y)
                y50 = min(list_alt)
                list_alt = []
            #--------------------------------Y75---------------------------------
            y75 = data[data['Z'] == z75]
            if y75.shape[0] > 0:
                y75 = y75['Y'].mean()
            else:
                for index, row in new_data.iterrows():
                    if row.Z > z75:
                        list_alt.append(row.Y)
                y75 = min(list_alt)
                list_alt = []
            #--------------------------------Y100--------------------------------
            y100 = data[data['Z'] == z100]
            if y100.shape[0] > 0:
                y100 = y100['Y'].mean()
            else:
                for index, row in x0.iterrows():
                    if row.Z < z100:
                        list_alt.append(row.Y)
                y100 = max(list_alt)
                list_alt = []
            #--------------------------------IF----------------------------------
            if(y0 < y25) and (y25 <= y50) and (y50 >= y75) and (y75 > y100):
                list_index.append(i)
                continue
            #--------------------------------------------------------------------
        print(f"Possíveis clusters: {list_index}")
        print('---------------------------------------------------------')
        return list_index
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def main(path, quadrante):
    #-----------------------Corte----------------------------
    data = pd.read_csv(path)
    new_data = data[['X', 'Y', 'Z', 'reflectivity']].copy()
    new_data = new_data[new_data.Y.abs() <= 40]
    #########################################################
    #-------------------1º Kmeans----------------------------
    processed_data = kmeans(new_data, 5)
    processed_data = getQuadrante(new_data, quadrante)
    filtered_data = filtro(processed_data)
    #########################################################
    #-------------------2º Kmeans----------------------------
    processed_data = wire_compare(processed_data)
    try:
        processed_data = kmeans(processed_data, 8)
        list_tree = find_tree(processed_data, N_CLUSTERS = 8)
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados processados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #########################################################
    #--------------------3º Kmeans---------------------------
    try:
        if(filtered_data.shape[0] > 0):
            #filtered_data = deleta_discrepantes(filtered_data, N_CLUSTERS = 8)
            #filtered_data = deleta_verticais(filtered_data, N_CLUSTERS = 8)
            if(len(filtered_data) < 4):
                print(f"Sem fios no quadrante da árvore!")
                print('---------------------------------------------------------')
            else:
                filtered_data = kmeans(filtered_data, 5)
        else:
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(f"Sem dados filtrados!")
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados filtrados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #########################################################
    soma10(new_data, filtered_data)
    #-------------------------Cálculos-----------------------
    max_index = getClusterSize(processed_data, N_CLUSTERS = 8)
    if list_tree:
        for index in max_index:
            if index in list_tree:
                max_index = index
                break
    else:
        max_index = max_index[0]
    print(f"Cluster Selecionado: {max_index}")
    soma10(new_data, processed_data[processed_data.cluster == max_index])
    #processed_data[processed_data.cluster == max_index].to_csv('/home/ubuntu/Downloads/filtro.csv')
    print('---------------------------------------------------------')
    try:
        distancia = getDistancia(processed_data, max_index)
        altura = getAltura(processed_data, max_index)
        try:
            dist_compare(filtered_data, processed_data, max_index)
        except:
            print('Sem fios detectados na captura!')
            print('---------------------------------------------------------')
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados no quadrante: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return None, None, None
    #########################################################
    return distancia, altura, processed_data

if __name__ == '__main__':
    #path = "fevereiro/20220201175111/20220201181545322832.lidar.csv"
    path = "fevereiro/20220201175111/20220201181508435555.lidar.csv"
    #path = "fevereiro/20220201175111/20220201181259479249.lidar.csv"
    quadrante = 'top-left'
    #path = "fevereiro/20220201175111/20220201181314067145.lidar.csv"
    #quadrante = 'top-right'
    distancia, altura, _ = main(path, quadrante)

    if distancia and altura:
        print(f"Altura: {altura:.2f} metros")
        print(f"Distancia: {distancia:.2f} metros")
        print('---------------------------------------------------------')