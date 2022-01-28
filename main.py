import cloud_kmeans
import quadrante_imagem
import cv2
import pandas as pd

class Main:
    def __init__(self, image, box) -> None:
        self._image = image
        self._box = box

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = cv2.imread(image)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        self._box = box

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, center):
        self._center = center

    @property
    def quadrante(self):
        return self._quadrante

    @quadrante.setter
    def quadrante(self, quadrante):
        self._quadrante = quadrante

    @property
    def altura(self):
        return self._altura

    @altura.setter
    def altura(self, altura):
        self._altura = altura

    @property
    def largura(self):
        return self._largura

    @largura.setter
    def largura(self, largura):
        self._largura = largura

    @property
    def distancia(self):
        return self._distancia

    @distancia.setter
    def distancia(self, distancia):
        self._distancia = distancia

    @property
    def pointCloud(self):
        return self._pointCloud

    @pointCloud.setter
    def pointCloud(self, pointCloud):
        self._pointCloud = pointCloud

    def __str__(self) -> str:
        return f"Quadrante: {self._quadrante}\nCentro: {self.center}"

if __name__ == '__main__':
    image = cv2.imread('20211210121331.camera.jpg')
    box = [[800, 2, 800, 2],[20, 20, 200, 200]]
    lidar = "new_csv/cap35.csv"

    newMain = Main(image, box)
    newMain.center = quadrante_imagem.getCenter(newMain.box)
    newMain.quadrante = quadrante_imagem.getQuadrante(newMain.image, newMain.center)

    print(newMain)
    print('---------------------------------------------------------')
    newMain.distancia, newMain.altura, newMain.pointCloud = cloud_kmeans.main(lidar, newMain.quadrante)

    print(f"Altura: {newMain.altura:.2f} metros")
    print(f"Distancia: {newMain.distancia:.2f} metros")
    print('---------------------------------------------------------')
    
