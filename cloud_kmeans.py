import os

from regex import I
import cloudPoints
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
import yaml

pd.options.mode.chained_assignment = None

def filtro(raw_points) -> pd.DataFrame:
    new_df : list= []
    for _, row in raw_points.iterrows():
        if (2 < row['Y'] < 10):
            if (-0.5 < row['Z'] < 3) and (3 < abs(row['X']) < 7):
                new_df.append(row)
        elif (10 < row['Y'] < 15):
            if (-1.75 < row['Z'] < 4.5) and (3 < abs(row['X']) < 7):
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
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXxxxxxxxxxxxxxxxxxxxxx')

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

def find_tree(data, N_CLUSTERS):
    list_alt, list_index = [], []
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

        print(f"Score: {score}")

        line_X = np.arange(vet_coord[i][0].min(), vet_coord[i][0].max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        vet_coord.append([vet_coord[i][0][inlier_mask], vet_coord[i][1][inlier_mask]])

        vet_test = filtered_data[filtered_data[['X','Y']].apply(tuple,1).isin(zip(list(vet_coord[i][0][inlier_mask].flat), list(vet_coord[i][1][inlier_mask].flat)))]

        X1 = vet_coord[i][0][inlier_mask]
        Y1 = vet_coord[i][1][inlier_mask]

        for k in range(len(X1) - 1):
            dist += math.sqrt((X1[k+1] - X1[k])**2 + (Y1[k+1] - Y1[k])**2)
        media = dist / len(X1)
        print(f"Média: {media:.2f}")

        '''plt.scatter(vet_coord[i][0][inlier_mask], vet_coord[i][1][inlier_mask], color = "yellowgreen", marker = ".", label = "Inliers")
        plt.scatter(vet_coord[i][0][outlier_mask], vet_coord[i][1][outlier_mask], color = "gold", marker = ".", label = "Outliers")

        plt.plot(line_X, line_y_ransac, color = "cornflowerblue", linewidth = 2, label = "RANSAC regressor",)
        plt.legend(loc = "lower right")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()'''
    
    return (filtered_data[filtered_data[['X','Y']].apply(tuple,1).isin(zip(list(vet_coord[i][0][inlier_mask].flat), list(vet_coord[i][1][inlier_mask].flat)))], score, media)

def testa_arvores(processed_data):
    try:
        dict_cluster, list_tree, potenciais_arvores, y_cluster = [], [], [], []
        for i in range(16):
            y_cluster.append(processed_data[processed_data.cluster == i]['Y'].mean())
        dist1 = 5 #distancia a ser determinada, para cluster de árvore próximo ao carro. Por exemplo, 5
        std_near = 1.1 # STD normalmente encontrado em árvores próximas ao carro

        list_tree = find_tree(processed_data, N_CLUSTERS = 16)
        for tree in list_tree:
            _, score_arvore, media_arvore = ransac(processed_data[processed_data.cluster == tree].copy(), N_TIMES = [0.25])
            dict_cluster.append({
                'Cluster': tree, 
                'STD_X_Value': processed_data[processed_data.cluster == tree]['X'].std(),
                'STD_Y_Value': processed_data[processed_data.cluster == tree]['Y'].std(),
                'STD_Z_Value': processed_data[processed_data.cluster == tree]['Z'].std(),
                'DIST_Value': getDistancia(processed_data, tree),
                'Y_mean': y_cluster[tree],
                'Media_Dist': media_arvore,
                'OVERALL': 0
            })
 
        for cluster in dict_cluster:
            std_adapted = 0
            if cluster["DIST_Value"] < dist1:
                std_adapted = std_near
            else: #a árvore está distante, então o STD pode ser menor. Digamos, 50% menor
                std_adapted = std_near * 0.5

            if cluster["STD_X_Value"] >= std_adapted:
                #classifica como árvore
                if cluster["Y_mean"] > 7:
                    potenciais_arvores.append(cluster)
                    cluster["OVERALL"] = cluster["STD_X_Value"]**2 / cluster["DIST_Value"]
                    if( 0.2 < cluster["Media_Dist"] < 0.35):
                        cluster["OVERALL"] += 100
            else:
                cluster["OVERALL"] = -1
            print(yaml.dump(cluster, default_flow_style=False))

        print("------------------------------------Potenciais árvores----------------------------------")
        potenciais_arvores = sorted(potenciais_arvores, key=lambda d: d['OVERALL'], reverse = True) 
        for elemento in potenciais_arvores:
            print(yaml.dump(elemento, default_flow_style=False))
        return potenciais_arvores[0]['Cluster'], score_arvore
    except Exception as e:
        print(f"Sem árvores encontradas: {e}")
        return 0, 0
    
def testa_fios(filtered_data):
    dict_fios, potenciais_fios = [], []
    for i in range(2):
        _, score_fio, media_fio = ransac(filtered_data[filtered_data.cluster == i].copy(), N_TIMES = [0.25])
        dict_fios.append({
            'Cluster': i,
            'Media_Dist': media_fio,
            'Score_Fio': score_fio,
            'OVERALL': 0
        })

    for cluster in dict_fios:
        if(cluster["Media_Dist"] > 0.35):
            cluster["OVERALL"] += 100
        else:
            cluster["OVERALL"] = -1
        potenciais_fios.append(cluster)
        #print(yaml.dump(cluster, default_flow_style=False))

    potenciais_fios = sorted(potenciais_fios, key=lambda d: d['OVERALL'], reverse = True)
    #for fio in potenciais_fios:
    #    print(yaml.dump(fio, default_flow_style=False))
    max_index = potenciais_fios[0]['Cluster']
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
    filtered_data = filtro(processed_data)
    filtered_data = kmeans(filtered_data, N_CLUSTERS = 2)
    filtered_data, potenciais_fios = testa_fios(filtered_data.copy())
    processed_data = processed_data[~processed_data['X'].isin(list(filtered_data.X))]
    #########################################################
    #-------------------2º Kmeans----------------------------
    try:
        processed_data = kmeans(processed_data, N_CLUSTERS = 16)
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados processados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #########################################################
    #--------------------3º Kmeans---------------------------
    try:
        if(filtered_data.shape[0] > 0):
            if(len(filtered_data) < 4):
                print(f"Sem fios no quadrante da árvore!")
                print('---------------------------------------------------------')
            else:
                filtered_data = kmeans(filtered_data, N_CLUSTERS = 16)
        else:
            print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(f"Sem dados filtrados!")
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    except Exception as e:
        print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"Sem dados filtrados: {e}")
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    soma10(new_data.copy(), filtered_data.copy())
    #-------------------------Cálculos-----------------------
    max_index, score_arvore = testa_arvores(processed_data.copy())   

    print('---------------------[main]------------------------------')
    print(f"-> Cluster Selecionado: {max_index}")
    print('---------------------------------------------------------')
    soma10(new_data.copy(), processed_data[processed_data.cluster == max_index].copy())

    frames = [processed_data[processed_data.cluster == max_index].copy(), filtered_data]
    df = pd.concat(frames)
    soma10(new_data.copy(), df.copy())
    
    try:
        distancia = getDistancia(processed_data, max_index)
        altura = getAltura(processed_data, max_index)
        try:
            contato = dist_compare(filtered_data, processed_data, max_index)
            print(f'Distância de contato entre a árvore e o fio: {contato:.2f} metros.')
        except:
            print('xxxxxxxxxxxxxxxxxxxxx[main]xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print('Sem fios detectados na captura!')
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
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