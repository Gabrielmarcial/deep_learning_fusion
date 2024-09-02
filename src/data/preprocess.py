import os
import numpy as np
from osgeo import gdal
from sklearn.preprocessing import MinMaxScaler

def load_tiff_image(image):
    """
    Abre dados com a gdal e disponibiliza em um tensor numpy
    """
    gdal_header = gdal.Open(image)
    img_gdal = gdal_header.ReadAsArray()
    img = np.transpose(img_gdal, (1,2,0))
    return img

def create_label_image(rgb_image, color2label):
    """
    Cria uma matriz (tensor de profundidade 1) partindo de um tensor RGB.
    Requer um dicionário que mapeie valores RGB em valores inteiros (supondo uso de sparse_categorical_crossentropy loss)
    """
    W = np.power(256, [[0],[1],[2]])
    img_index = rgb_image.dot(W).squeeze(-1)
    values = np.unique(img_index)
    label_image = np.zeros(img_index.shape)
    for i, c in enumerate(values):
        try:
            label_image[img_index==c] = color2label[tuple(rgb_image[img_index==c][0])]
        except:
            pass
    return label_image

def normalization(image):
    """
    Normaliza um tensor numpy para valores reais no intervalo (0,1)
    """
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized_ = scaler.fit_transform(image_reshaped)
    image_normalized = image_normalized_.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized

""" Possivelmente vamos isolar o preprocessamento do treinamento mais adiante
def preprocess_data(raw_data_dir, processed_data_dir):
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    for subdir in os.listdir(raw_data_dir):
        subdir_path = os.path.join(raw_data_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                # Processar cada imagem aqui, se necessário
                # Exemplo: converter imagem em array numpy e aplicar transformações
                # Salvar imagem processada no diretório de dados processados

if __name__ == '__main__':
    preprocess_data('data/raw', 'data/processed')
"""