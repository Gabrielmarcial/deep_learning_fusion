import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf

# Function: extract_patches
# -------------------------
# Extract patches from the original and reference image
#
# Input parameters:
#   image      = array containing the original image (h,w,c)
#   reference  = array containing the reference image (h,w,c)
#   patch_size = patch size (scalar). The shape of the patch is square.
#   stride     = displacement to be applied.
#   border_patches = include patches overlaping image borders
#
# Returns:
#   A, B = List containing the patches for the input image (A) and respective reference (B).
#
def extract_patches(image, reference, patch_size, stride, 
                    border_patches=False, image_channels = 3):
    """
    Rotina dos materiais de aula do prof. Gilson para pré-processamento.
    """
    print("Image dimensions:", image.shape)
    print("Reference dimensions:", reference.shape)

    patch_img = []
    patch_ref = []

    h = image.shape[0] // stride
    w = image.shape[1] // stride

    for m in range(0, h):
        for n in range(0, w):
            #print('M %d, N %d, start %d finish %d , start %d finish %d' % (m, n, m*stride , m*stride+patch_size, n*stride, n*stride+patch_size) )
            if ( (m*stride+patch_size <= image.shape[0]) and (n*stride+patch_size <= image.shape[1]) ):
                patch_img.append( image[m*stride:m*stride+patch_size,n*stride:n*stride+patch_size,:] )
                patch_ref.append( reference[m*stride:m*stride+patch_size,n*stride:n*stride+patch_size] )
            elif border_patches:
                border_patch_img = np.zeros((patch_size,patch_size,image_channels))
                border_patch_ref = np.zeros((patch_size,patch_size))
                if (m*stride+patch_size > image.shape[0]):
                  border_mmax = patch_size-(m*stride+patch_size-image.shape[0])
                else:
                  border_mmax = patch_size-1
                if (n*stride+patch_size > image.shape[1]):
                  border_nmax = patch_size-(n*stride+patch_size-image.shape[1])
                else:
                  border_nmax = patch_size-1

                border_patch_img[0:border_mmax,0:border_nmax,:] = image[m*stride:m*stride+border_mmax,n*stride:n*stride+border_nmax,:]
                border_patch_ref[0:border_mmax,0:border_nmax] = reference[m*stride:m*stride+border_mmax,n*stride:n*stride+border_nmax]
                patch_img.append( border_patch_img )
                patch_ref.append( border_patch_ref )

    return np.array(patch_img), np.array(patch_ref)

"""
def split_data(processed_data_dir, splits_dir, test_size=0.2, val_size=0.2):
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    for subdir in os.listdir(processed_data_dir):
        subdir_path = os.path.join(processed_data_dir, subdir)
        if os.path.isdir(subdir_path):
            images = [os.path.join(subdir_path, img) for img in os.listdir(subdir_path)]
            train_val, test = train_test_split(images, test_size=test_size, random_state=42)
            train, val = train_test_split(train_val, test_size=val_size, random_state=42)

            for dataset, dataset_name in zip([train, val, test], ['train', 'val', 'test']):
                dataset_dir = os.path.join(splits_dir, dataset_name, subdir)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                for img in dataset:
                    shutil.copy(img, dataset_dir)

if __name__ == '__main__':
    split_data('data/processed', 'data/splits')
"""
