import pandas as pd
import os

packages = "/media/nvidia/SSD/Source/utils/"
sys.path.insert(0, packages)

from log import Log
log = Log("main-capture-Lidar")

class Lidar:
    def __init__(self):
        pass

    @property
    def cloud(self):
        return self._cloud

    @cloud.setter
    def cloud(self, cloud):
        self._cloud = pd.read_csv(cloud)

    def getProperty(self, property):
        return self._cloud[property]

    def deleteProperty(self, property):
        self._cloud = self._cloud.drop(property, axis = 1)
        log.info(f"[deleteProperty] - Removendo a propriedade {property}")

    def saveCSV(self, path):
        self._cloud.to_csv(path)
        log.info(f"[saveCSV] - Salvando o arquivo {path}")

    def deleteCSV(self, path):
        os.system(f"rm {path}")
        log.info(f"[deleteCSV] - Removendo o arquivo {path}")

    def filterDF(self):
        new_df = []
        for index, row in self._cloud.iterrows():
            if row['intensity'] > 0 :
                #row['Points_m_XYZ:2'] += 30
                new_df.append(row)
        new_df = pd.DataFrame(new_df)
        frames = pd.concat[self._cloud, new_df]
        log.info(f"[filterDF] - Filtrando o dataframe. Filtro: intensidade > 0")
        return new_df

