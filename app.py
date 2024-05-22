import numpy as np
import cv2
import matplotlib.pyplot as plt 

class ElementMatrix:
    def __init__(self, matrix, pos_x, pos_y):
        self.matrix = matrix
        self.origin = (pos_x, pos_y)

def dilation(image, elem_matrix):
    img_height, img_width = image.shape
    elem_height, elem_width = elem_matrix.matrix.shape
    new_image = np.ones((img_height, img_width))

    for i in range(img_height - elem_height + 1):
        for j in range(img_width - elem_width + 1):
            if image[i + elem_matrix.origin[0], j + elem_matrix.origin[1]] == 0:
                for x in range(i, i + elem_height):
                    for y in range(j, j + elem_width):
                        if elem_matrix.matrix[x - i, y - j] == 0:
                            new_image[x, y] = elem_matrix.matrix[x - i, y - j]
    return new_image

def erosion(image, elem_matrix):
    img_height, img_width = image.shape
    elem_height, elem_width = elem_matrix.matrix.shape
    new_image = np.ones((img_height, img_width))
    for i in range(img_height - elem_height + 1):
        for j in range(img_width - elem_width + 1):
            flag = True                
            for x in range(i, i + elem_height):
                for y in range(j, j + elem_width):
                    if elem_matrix.matrix[x - i, y - j] == 0:
                        if image[x, y] != 0:
                            flag = False
            if flag:
                new_image[i + elem_matrix.origin[0]][j + elem_matrix.origin[1]] = 0 
    return new_image

def binarize_image(image):
    img_height, img_width = image.shape
    new_image = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            if image[i, j] >= 128:   
                new_image[i, j] = 1
    return new_image

def make_image_radable_by_cv2(image):
    img_height, img_width = image.shape
    new_image = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            if image[i, j] == 0:
                new_image[i, j] = 0
            else:
                new_image[i, j] = 255
    return new_image

if __name__ == "__main__":
    cruz = np.array([[1,0,1],[0,0,0],[1,0,1]])  
    diamante = np.array([[1,1,0,1,1],[1,0,0,0,1],[0,0,0,0,0],[1,0,0,0,1],[1,1,0,1,1]])
    diamante7 = np.array([[1,1,1,0,1,1,1],[1,1,0,0,0,1,1],[1,0,0,0,0,0,1],[0,0,0,0,0,0,0],[1,0,0,0,0,0,1],[1,1,0,0,0,1,1],[1,1,1,0,1,1,1]])
    vertical_bar = np.array([[0],[0],[0],[0],[0]])
    horizontal_bar = np.array([[0,1,0,1,0]])

    cruz_ct_x = 1
    cruz_ct_y = 1

    diamante_ct_x = 2
    diamante_ct_y = 2
    
    struct_elem_cruz = ElementMatrix(cruz, cruz_ct_x, cruz_ct_y)
    struct_elem_diamante = ElementMatrix(diamante, diamante_ct_x, diamante_ct_y)
    struct_elem_diamante7 = ElementMatrix(diamante7, 3, 3)
    struct_elem_vertical_bar = ElementMatrix(vertical_bar, 0 , 0)
    struct_elem_horizontal_bar = ElementMatrix(horizontal_bar, 0, 2)


    original_image = cv2.imread("figuraV2.jpg", cv2.IMREAD_GRAYSCALE)
    image = binarize_image(original_image)
    dilated_img = dilation(image, struct_elem_horizontal_bar)

    res = make_image_radable_by_cv2(dilated_img)
    cv2.imwrite('result.png', res)

    plt.imshow(res, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()


    
