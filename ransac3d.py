import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/ubuntu/Downloads/lidar/fevereiro/20220201175111/20220201175846172005.lidar.csv")

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])