import cv2

def getCenter(box):
    if(box == -1):
        return [0, 0]
    x_minimo = abs(int(box[0][0] - (box[0][2] / 2)))
    if (x_minimo - 15) >= 0:
        x_minimo = x_minimo - 15
    x_maximo = abs(int(box[0][0] + (box[0][2] / 2))) + 15
    y_minimo = abs(int(box[0][1] - (box[0][3] / 2)))
    if (y_minimo - 15) >= 0:
        y_minimo = y_minimo - 15
    y_maximo = abs(int(box[0][1] + (box[0][3] / 2))) + 15

    x_medio = (x_maximo - x_minimo) / 2
    y_medio = (y_maximo - y_minimo) / 2

    return [x_medio, y_medio]

def getQuadrante(image, center):
    height = image.shape[0]
    width = image.shape[1]
    mid_height = height / 2
    mid_width = width / 2
    
    if(center[0] == 0) and (center[1] == 0):
        quadrante = 'Null'
    elif(center[0] > mid_width) and (center[1] > mid_height):
        quadrante = 'top-right'
    elif(center[0] > mid_width) and (center[1] < mid_height):
        quadrante = 'bottom-right'
    elif(center[0] < mid_width) and (center[0] > mid_height):
        quadrante = 'top-left'
    elif(center[0] < mid_width) and (center[0] < mid_height):
        quadrante = 'bottom-left'
    return quadrante

if __name__ == '__main__':
    path = '20211210121331.camera.jpg'
    image = cv2.imread(path)
    box = [[10, 10, 100, 100],[20, 20, 200, 200]]

    center = getCenter(box)
    print(f"Centro: {center}")
    quadrante = getQuadrante(image, center)
    print(f"Quadrante: {quadrante}")

