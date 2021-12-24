# Veloview scripting

Veloview Python script that:
- Loads VLP-16 calibration file
- Connects to the ethernet port
- Capture the point cloud
- Save the point cloud as a .csv.

Python3 script that:
- Kmeans the point cloud
- Find the distance and de height from the mean of the largest kmeans cluster
- Find the center of the yolo bbox in a image and compare with the center of the largest kmeans cluster
