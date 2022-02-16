from sklearn import linear_model
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def ransac(X, y):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    print(ransac.estimator_.coef_)

    lw = 2
    plt.scatter(
        X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
    )

    plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=lw, label="RANSAC regressor",)
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()

if __name__ == '__main__':
    path = "fevereiro/20220201175111/20220201175353660589.lidar.csv"
    data = pd.read_csv(path)
    new_data = data[['X', 'Y', 'Z', 'reflectivity']].copy()
    new_data = new_data[new_data.Y.abs() <= 40]
    corte_x = (abs(new_data['X'].mean())) + (abs(new_data['X'].std()))
    new_data = new_data[new_data.X.abs() <= corte_x]

    X = new_data['X'].to_numpy().reshape(-1, 1)
    Y = new_data['Y'].to_numpy().reshape(-1, 1)
    ransac(X, Y)