import cloudPoints
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml

from sklearn import linear_model
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None

def filtro(raw_points) -> pd.DataFrame:
    new_df : list= []
    for _, row in raw_points.iterrows():
        if (1 < row['Y'] < 10):
            if (0 < row['Z'] < 4.5) and (1 < abs(row['X']) < 12):
                new_df.append(row)
        elif (10 < row['Y'] < 15):
            if (-1 < row['Z'] < 4.5) and (1 < abs(row['X']) < 12):
                new_df.append(row)  
    return pd.DataFrame(new_df)

def filtro_arvore(raw_points) -> pd.DataFrame:
    new_df : list= []
    for _, row in raw_points.iterrows():
        if (0 < row['Y'] < 10):
            if (0 < row['Z'] < 4.5):
                new_df.append(row)
        elif (10 < row['Y']):
            if (-2 < row['Z'] < 4.5):
                new_df.append(row) 
    return pd.DataFrame(new_df)

def plot(data):
    cloudPoints.generate_mesh(cloudPoints.slam([data], [[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0]]))

def kmeans(new_data, N_CLUSTERS):
    new_data['cluster'] = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++',  random_state = 42).fit_predict(new_data.iloc[:, 0:3].values)
    return new_data

def getDistancia(processed_data, max_index):
    distX, distY, distZ, distT = [], [], [], []
    new_data = processed_data[processed_data.cluster == max_index]
    try:
        for _, row in new_data.iterrows():
            distX.append(row.X)
            distY.append(row.Y)
            distZ.append(row.Z)
        for i in range(len(distX)):
            distT.append(math.sqrt(distX[i]**2 + distY[i]**2 + distZ[i]**2))
        return (min(distT))
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxx[getDistancia]xxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def getAltura(data, max_index):
    new_data = data[data.cluster == max_index]
    try:
        altura = new_data['Z'].loc[new_data['Z'].idxmax()] - new_data['Z'].mean()
        if(new_data['Z'].std() > 1):
            altura = altura / math.sqrt(new_data['Z'].std())
        return altura
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxx[getAltura]xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def getQuadrante(data, side):
    try:
        if(side == 'top-right'):
            return (data[data['X'] > 0])
        elif(side == 'top-left'):
            return (data[data['X'] < 0])
        elif(side == 'bottom-right'):
            return (data[data['X'] > 0])
        elif(side == 'bottom-left'):
            return (data[data['X'] < 0])
        elif(side == 'Null'):
            return data
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxx[getQuadrante]xxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def dist_compare(filtered_data, processed_data, max_index):
    distX, distY, distZ, distT = [], [], [], []
    x, y, z = [], [], []
    new_data = processed_data[processed_data.cluster == max_index]
    x1, y1, z1 = new_data['X'], new_data['Y'], new_data['Z']

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
        distT.append(math.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
    return min(distT)

def soma10(df, new_filtered_data):
    new_filtered_data['Z'] = new_filtered_data['Z'] + 10
    plot(pd.concat([df, new_filtered_data]))

def aux_find_tree(data, Z, new_data):
    list_alt = []
    X = data[data['Z'] == Z]
    if X.shape[0] > 0:
        X = X['X'].mean()
    else:
        for _, row in new_data.iterrows():
            if row.Z > Z:
                list_alt.append(row.X)
        X = min(list_alt)
    return X

def find_tree(data, N_CLUSTERS):
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
            x0 = aux_find_tree(data, z0, new_data)
            x25 = aux_find_tree(data, z25, new_data)
            x50 = aux_find_tree(data, z50, new_data)
            x75 = aux_find_tree(data, z75, new_data)
            x100 = aux_find_tree(data, z100, new_data)
            #--------------------------------IF----------------------------------
            if(x0 > x25) and (x25 >= x50) and (x50 <= x75) and (x75 < x100):
                list_index.append(i)
                continue
            #--------------------------------X0----------------------------------
            x0 = aux_find_tree(data, z0, new_data)
            x25 = aux_find_tree(data, z25, new_data)
            x50 = aux_find_tree(data, z50, new_data)
            x75 = aux_find_tree(data, z75, new_data)
            x100 = aux_find_tree(data, z100, new_data)
            #--------------------------------IF----------------------------------
            if(x0 < x25) and (x25 <= x50) and (x50 >= x75) and (x75 > x100):
                list_index.append(i)
                continue
            #--------------------------------Y0----------------------------------
            y0 = aux_find_tree(data, z0, new_data)
            y25 = aux_find_tree(data, z25, new_data)
            y50 = aux_find_tree(data, z50, new_data)
            y75 = aux_find_tree(data, z75, new_data)
            y100 = aux_find_tree(data, z100, new_data)
            #--------------------------------IF----------------------------------
            if(y0 > y25) and (y25 >= y50) and (y50 <= y75) and (y75 < y100):
                list_index.append(i)
                continue
            #--------------------------------Y0----------------------------------
            y0 = aux_find_tree(data, z0, new_data)
            y25 = aux_find_tree(data, z25, new_data)
            y50 = aux_find_tree(data, z50, new_data)
            y75 = aux_find_tree(data, z75, new_data)
            y100 = aux_find_tree(data, z100, new_data)
            #--------------------------------IF----------------------------------
            if(y0 < y25) and (y25 <= y50) and (y50 >= y75) and (y75 > y100):
                list_index.append(i)
                continue
            #--------------------------------------------------------------------
        print('--------------------[find_tree]-----------------------------')
        print(f"-> Possíveis clusters: {list_index}")
        print('------------------------------------------------------------')
        return list_index
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxx[find_tree]xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def ransac(filtered_data, N_TIMES):
    vet_coord = [[filtered_data['X'].to_numpy().reshape(-1, 1), filtered_data['Y'].to_numpy().reshape(-1, 1)]]
    for i in range(len(N_TIMES)):
        dist = 0
        
        ransac = linear_model.RANSACRegressor(residual_threshold = N_TIMES[i]).fit(vet_coord[i][0], vet_coord[i][1])
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        score = ransac.score(vet_coord[i][0], vet_coord[i][1])

        line_X = np.arange(vet_coord[i][0].min(), vet_coord[i][0].max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        vet_coord.append([vet_coord[i][0][inlier_mask], vet_coord[i][1][inlier_mask]])

        X1, Y1 = vet_coord[i][0][inlier_mask], vet_coord[i][1][inlier_mask]

        for k in range(len(X1) - 1):
            dist += math.sqrt((X1[k+1] - X1[k])**2 + (Y1[k+1] - Y1[k])**2)
        media = dist / len(X1)

        '''plt.scatter(vet_coord[i][0][inlier_mask], vet_coord[i][1][inlier_mask], color = "yellowgreen", marker = ".", label = "Inliers")
        plt.scatter(vet_coord[i][0][outlier_mask], vet_coord[i][1][outlier_mask], color = "gold", marker = ".", label = "Outliers")

        plt.plot(line_X, line_y_ransac, color = "cornflowerblue", linewidth = 2, label = "RANSAC regressor",)
        plt.legend(loc = "lower right")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()'''
    
    return (filtered_data[filtered_data[['X','Y']].apply(tuple,1).isin(zip(list(vet_coord[i][0][inlier_mask].flat), list(vet_coord[i][1][inlier_mask].flat)))], score, media)

def testa_arvores(processed_data, fio_data):
    try:
        dict_cluster, list_tree, potenciais_arvores, y_cluster, x_cluster = [], [], [], [], []
        for i in range(15):
            y_cluster.append(processed_data[processed_data.cluster == i]['Y'].mean())
            x_cluster.append(processed_data[processed_data.cluster == i]['X'].mean())
        list_tree = find_tree(processed_data, N_CLUSTERS = 15)
        for tree in list_tree:
            _, score_arvore, media_arvore = ransac(processed_data[processed_data.cluster == tree].copy(), N_TIMES = [0.25])

            vet_coord = [processed_data[processed_data.cluster == tree]['X'].to_numpy().reshape(-1, 1), processed_data[processed_data.cluster == tree]['Y'].to_numpy().reshape(-1, 1)]
            X1, Y1 = vet_coord[0], vet_coord[1]

            dist = []
            for k in range(len(X1) - 1):
                arr = round(math.sqrt((X1[k+1] - X1[k])**2 + (Y1[k+1] - Y1[k])**2), 2)
                dist.append(arr)
            media_arvore = max(set(dist), key=dist.count)

            dict_cluster.append({
                'Cluster': tree, 
                'STD_X_Value': processed_data[processed_data.cluster == tree]['X'].std(),
                'STD_Y_Value': processed_data[processed_data.cluster == tree]['Y'].std(),
                'STD_Z_Value': processed_data[processed_data.cluster == tree]['Z'].std(),
                'DIST_Value': getDistancia(processed_data, tree),
                'Y_mean': y_cluster[tree],
                'X_mean': x_cluster[tree],
                'Media_Dist': media_arvore,
                'OVERALL': 0
            })

        for cluster in dict_cluster:
            print(f"Media_Dist: {cluster['Media_Dist']}")
            if cluster["Y_mean"] > 1:
                potenciais_arvores.append(cluster)
                cluster["OVERALL"] = (1 / cluster["DIST_Value"])
                if( 0.22 <= cluster["Media_Dist"] <= 0.43):
                    cluster["OVERALL"] += 100  #+ cluster["Media_Dist"]
            print(f"OVERALL: {cluster['OVERALL']}")

        print("------------------------------------Potenciais árvores----------------------------------")
        potenciais_arvores = sorted(potenciais_arvores, key = lambda d: d['OVERALL'], reverse = True) 
        for elemento in potenciais_arvores:
            print(yaml.dump(elemento, default_flow_style = False))
        return potenciais_arvores[0]['Cluster'], score_arvore
    except Exception as e:
        print(f"Sem árvores encontradas: {str(e)}")
        return -1, -1
    
def testa_fios(filtered_data):
    dict_fios, potenciais_fios = [], []
    for i in range(2):
        _, score_fio, media_fio = ransac(filtered_data[filtered_data.cluster == i].copy(), N_TIMES = [0.25])

        vet_coord = [filtered_data[filtered_data.cluster == i]['X'].to_numpy().reshape(-1, 1), filtered_data[filtered_data.cluster == i]['Y'].to_numpy().reshape(-1, 1)]
        X1, Y1 = vet_coord[0], vet_coord[1]

        dist = []
        for k in range(len(X1) - 1):
            arr = round(math.sqrt((X1[k+1] - X1[k])**2 + (Y1[k+1] - Y1[k])**2), 2)
            dist.append(arr)
        media_fio = max(set(dist), key=dist.count)
        print(f"Media_Dist_Fio: {media_fio}")

        dict_fios.append({
            'Cluster': i,
            'Media_Dist': media_fio,
            'Score_Fio': score_fio,
            'OVERALL': 0
        })
    for cluster in dict_fios:
        test_data = filtered_data[filtered_data.cluster == cluster['Cluster']]
        maior_z = test_data['Z'].loc[test_data['Z'].idxmax()]
        maior_y = test_data.loc[test_data['Z'] == test_data['Z'].loc[test_data['Z'].idxmax()]].iloc[0]['Y']
        if (maior_y > 10) and (maior_z > 3):
            overall = 100
        elif ((maior_y > 5) and (maior_y < 10)) and (maior_z > 2):
            overall = 80
        elif ((maior_y > 2) and (maior_y < 5)) and (maior_z > 1):
            overall = 60
        else:
            overall = 40
 
        if(cluster["Media_Dist"] > 0.35):
            cluster["OVERALL"] += 100 + cluster["Media_Dist"] + overall
        else:
            cluster["OVERALL"] = cluster["Media_Dist"] + overall
        potenciais_fios.append(cluster)
        print(f"OVERALL FIO: {cluster['OVERALL']}")

    potenciais_fios = sorted(potenciais_fios, key=lambda d: d['OVERALL'], reverse = True)
    max_index = potenciais_fios[0]['Cluster']
    maior_z = filtered_data['Z'].loc[filtered_data['Z'].idxmax()]
    maior_y = filtered_data.loc[filtered_data['Z'] == filtered_data['Z'].loc[filtered_data['Z'].idxmax()]].iloc[0]['Y']
    print(f"Media_Dist do Fio: {dict_fios[max_index]['Media_Dist']}")
    return filtered_data[filtered_data.cluster == max_index], potenciais_fios

def main(path, quadrante):
    #-----------------------Corte----------------------------
    data = pd.read_csv(path)
    new_data = data[['X', 'Y', 'Z']].copy()
    new_data = new_data[new_data.Y > 0]
    new_data = new_data[new_data.Y < 19]
    new_data = new_data[new_data.X > 0]
    new_data = new_data[new_data.X < 25]
    #########################################################
    #-------------------1º Kmeans----------------------------
    processed_data = getQuadrante(new_data, quadrante)
    processed_data = filtro_arvore(new_data)
    filtered_data = filtro(processed_data)
    filtered_data = kmeans(filtered_data, N_CLUSTERS = 2)
    filtered_data, potenciais_fios = testa_fios(filtered_data.copy())
    processed_data = processed_data[~processed_data['X'].isin(list(filtered_data.X))]
    #########################################################
    #-------------------2º Kmeans----------------------------
    try:
        processed_data = kmeans(processed_data, N_CLUSTERS = 15)
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados processados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #########################################################
    #-------------------------Cálculos-----------------------
    max_index, score_arvore = testa_arvores(processed_data.copy(), filtered_data.copy()) 
    if max_index == -1:
        return 0, 0, 0, 0, 0  
    print('---------------------[main]------------------------------')
    print(f"-> Cluster Selecionado: {max_index}")
    print('---------------------------------------------------------')

    frames = [processed_data[processed_data.cluster == max_index].copy(), filtered_data]
    df = pd.concat(frames)
    soma10(new_data.copy(), filtered_data.copy())
    soma10(new_data.copy(), processed_data[processed_data.cluster == max_index].copy())
    soma10(new_data.copy(), df.copy())
    
    try:
        distancia, altura = getDistancia(processed_data, max_index), getAltura(processed_data, max_index)
        contato = dist_compare(filtered_data, processed_data, max_index)
        print(f'Distância de contato entre a árvore e o fio: {contato:.2f} metros.')
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados no quadrante: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return 0, 0, 0, 0, 0
    #########################################################
    return distancia, altura, processed_data, score_arvore, potenciais_fios[0]['Score_Fio']

def find_files():
	from os import listdir
	from os.path import isfile, join
	lidar_files = []
	onlyfiles = [f for f in listdir('/home/ubuntu/Downloads/lidar/202201281/20220128181215/') if isfile(join('/home/ubuntu/Downloads/lidar/202201281/20220128181215/', f))]
	for file in onlyfiles:
		if 'csv' in file:
			lidar_files.append(file)

	return lidar_files

if __name__ == '__main__':
    files = find_files()
    for file in files:
        try:
            input('Enter: ')
            os.system('clear')
            print('-----------------------[captura]---------------------------')
            path = (f"/home/ubuntu/Downloads/lidar/202201281/20220128181215/{file}")
            print('------------------------------------------------------------')
            print(path)
            quadrante = 'top-right'
            distancia, altura, _, score_arvore, score_fio = main(path, quadrante)

            if distancia and altura:
                print('-----------------------[__main__]---------------------------')
                print(f"-> Altura: {altura:.2f} metros")
                print(f"-> Distancia: {distancia:.2f} metros")
                print(f"-> Score_arvore: {score_arvore}")
                print(f"-> score_fio: {score_fio}")
                print('------------------------------------------------------------')
				
        except Exception as e:
            print(e)