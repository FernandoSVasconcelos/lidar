from typing import Dict, List
import pandas as pd
import numpy as np
import numba
import math
import open3d as o3d
import os


def generate_image(points : List[float], path : str) -> None:
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.add_geometry(pcd)
	ctr = vis.get_view_control()
	ctr.rotate(80.0, 120.0)
	vis.update_geometry(pcd)
	vis.poll_events()
	vis.update_renderer()
	vis.capture_screen_image(path)
	
	#vis.destroy_window()

def generate_mesh(points : List[float]) -> None:
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	
	'''
	pcd.estimate_normals(fast_normal_computation=True)
	
	distances = pcd.compute_nearest_neighbor_distance()
	avg_dist = np.mean(distances)
	radius = 15 * avg_dist
	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
	dec_mesh = mesh.simplify_quadric_decimation(100000)
	dec_mesh.remove_degenerate_triangles()
	dec_mesh.remove_duplicated_triangles()
	dec_mesh.remove_duplicated_vertices()
	dec_mesh.remove_non_manifold_edges()
	'''

	o3d.visualization.draw_geometries([pcd], point_show_normal = True)

def __isInsideRange(azimuth : float, range : List[float], distance : float, max_distance) -> bool:
    if(range[0] > range[1]):  # Caso de volta passando pelo angulo 0
        if((azimuth > range[0] and azimuth < 360 and distance < max_distance) or (azimuth > 0 and azimuth < range[1] and distance < max_distance)): 
            return True
    if(azimuth > range[0] and azimuth < range[1] and distance < max_distance):
        return True
    return False


def filterPoints(dataframe : List[float], angles : float, generate_meshs, max_distance : float) -> List[float]:
    dataframe['Azimuth'] = dataframe['Azimuth']/100
    dataframe = dataframe[dataframe.apply(lambda x : __isInsideRange(x['Azimuth'], angles, x['Distance'], max_distance) == True, axis=1)]
    dataframe.sort_values(by=['Azimuth'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def getProjection(points : List[float], K : float) -> List[float]:
	"""
	Realiza a projeção de matrizes.
	"""
	projection_points : List[float] = []
	for point in points:
		p = np.array(point)
		#r = np.matmul(p, K)
		nr = __vetMulMat(p, K)
		projection_points.append(nr)
	return projection_points

def convert3dto2d(points : List[float]) -> List[float]:
	"""
	Converte pontos 3d em pontos 2d.
	"""
	newPoints : List[float]= []
	for point in points:
		if(point[2] != 0):
			x = point[0]/(point[2])
			y = point[1]/(point[2])
			newPoints.append([x,y])
		else:
			newPoints.append([0,0])
	return newPoints

@numba.jit(nopython=True)
def __vetMulMat (A : float, B : float) -> float:
	"""
	Multiplica o vetor A pela matriz B.
	"""
	vec_size = len(A)
	C = np.zeros(vec_size) 
	for i in range(vec_size):
		for k in range(vec_size):
			C[i] += A[k] * B[i][k]
	return C

def rotateAxisX(points : List[float], th : float) -> List[float]:
	"""
	Rotaciona o eixo X.
	"""
	th = th * 0.0175 # radianos
	#Rotacao em X
	T=[[1,           0,              0      ],
	   [0,     math.cos(th),    math.sin(th)],
	   [0,      -math.sin(th),   math.cos(th)]]
	result : List[float] = []
	for point in points:
		p = np.array(point)
		#r = np.matmul(p, T)
		nr = __vetMulMat(p, np.array(T))
		#print('nr: ', nr)
		result.append(nr)
	return result


def rotateAxisZ(points : List[float], th : float) -> List[float]:
	"""
	Rotaciona o eixo Z.
	"""
	th = th * 0.0175 # radianos
	#Rotacao em Z
	T = [[math.cos(th),      math.sin(th),        0],        
		[-math.sin(th),      math.cos(th),        0],
		[     0,                  0,             1]]
	result = []
	for point in points:
		p = np.array(point)
		#r = np.matmul(p, T)
		nr = __vetMulMat(p, np.array(T))
		result.append(nr)
	return result

def slam(df : Dict, gps : List[str], giroscopio : List[str]) -> List[float]:
	if(len(df) == 1):
		xs = df[0]['Points_m_XYZ:0'].to_numpy()
		ys = df[0]['Points_m_XYZ:1'].to_numpy()
		zs = df[0]['Points_m_XYZ:2'].to_numpy()
		return map(list, zip(xs, ys, zs))

	xs = df[0]['Points_m_XYZ:0'].to_numpy()
	ys = df[0]['Points_m_XYZ:1'].to_numpy()
	zs = df[0]['Points_m_XYZ:2'].to_numpy()
	
	points : List[float] = []
	points.append(list(zip(xs, ys, zs)))

	flat_points = [item for sublist in points for item in sublist]
	return map(list, flat_points)

def savePoints2CSV(path : str, name : str, points : List[float], colors : List[int]) -> None:
	"""
	Salva os pontos 3d em um arquivo csv.
	"""

	unzip = list(zip(*points)) 

	if(colors == []): 
		r, g, b = [[],[],[]]
		while len(r) < len(unzip[0]): r.append(0)
		while len(g) < len(unzip[0]): g.append(0)
		while len(b) < len(unzip[0]): b.append(0)
	else:
		r, g, b = list(zip(*colors))

	d = {'X': list(unzip[0]), 'Y':list(unzip[1]), 'Z':list(unzip[2]), 'R': list(r), 'G':list(g), 'B': list(b)}


	csv = pd.DataFrame(data=d)
	csv.to_csv(path + name)

def save2dPoints2CSV(path : str, name : str, points : List[float]) -> None:
	"""
	Salva os pontos 2d em um arquivo csv.
	"""

	unzip = list(zip(*points)) 

	r, g, b, Z = [[], [], [], []]
	while len(r) < len(unzip[0]): r.append(0)
	while len(g) < len(unzip[0]): g.append(0)
	while len(b) < len(unzip[0]): b.append(0)
	while len(Z) < len(unzip[0]): Z.append(0)

	d = {'X': list(unzip[0]), 'Y':list(unzip[1]), 'Z':Z, 'R': r, 'G':g, 'B': b}

	csv = pd.DataFrame(data=d)
	csv.to_csv(path + name)
 
def filtro(raw_points):
    new_df = []
    for index, row in raw_points.iterrows():
        if row['intensity'] > 0 :
            #row['Points_m_XYZ:2'] += 30
            new_df.append(row)
    new_df = pd.DataFrame(new_df)
    frames = [raw_points, new_df]
    return new_df

def find_files():
	from os import listdir
	from os.path import isfile, join
	lidar_files = []
	onlyfiles = [f for f in listdir('/home/ubuntu/Downloads/new_csv/') if isfile(join('/home/ubuntu/Downloads/new_csv/', f))]
	for file in onlyfiles:
		if 'csv' in file:
			lidar_files.append(file)

	return lidar_files

if __name__ == '__main__':

	#points = slam([pd.read_csv('/mnt/SSD/Source/main/process/utils/reuse-data/lidar/2021-06-11.10:15:11.572066.velodyne.csv')], [[0,0], [0,0]], [[0,0,0], [0,0,0]])
	'''raw_points = pd.read_csv('cap1.csv')
	filtered_df = filtro(raw_points)
	points = slam([filtered_df], [[0,0], [0,0]], [[0,0,0], [0,0,0]])
	generate_mesh(points)'''
	files = find_files()
	for file in files:
		try:
			raw_points = pd.read_csv(f"/home/ubuntu/Downloads/new_csv/{file}")
			if raw_points is not None:
				#os.system(f"eog lidar_teste/2021121415/{file.split('.')[0]}.camera.jpg")
				
				filtered_df = filtro(raw_points)
				points = slam([filtered_df], [[0,0], [0,0]], [[0,0,0], [0,0,0]])
			
				generate_mesh(points)
				
		except Exception as e:
			print(e)
