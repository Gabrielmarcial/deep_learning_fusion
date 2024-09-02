import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.models.resunet import ResUNet
from src.data.preprocess import *
from scipy.stats import mode
from osgeo import gdal

def predict_image(model, image_path, patch_size):
	image = np.load(image_path)
	if image.shape != (patch_size,patch_size,4):
		return None	
	net_input = image/255
	predicted_classes = model.predict(net_input.reshape((1,patch_size,patch_size,4)))
	predicted = np.argmax(predicted_classes, axis=-1)
	return predicted[0], net_input[:,:,:3]

def plot_pred(reference, y, h, filename):
	if type(reference) == str:
		gt = np.load(reference)
	else:
		gt = reference
	fig1, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 10))
	ax1.imshow( y, cmap='gray', vmin=0, vmax=2 )
	ax1.set_title('Prediction (y)', fontsize=20)
	ax1.axis('off')
	ax2.imshow( gt, cmap='gray', vmin=0, vmax=2 )
	ax2.set_title('Ground truth (gt)', fontsize=20)
	ax2.axis('off')
	ax3.imshow( gt-y, cmap='gray', vmin=-1, vmax=1 )
	ax3.set_title('Diff', fontsize=20)
	ax3.axis('off')
	ax4.imshow( np.concatenate((h,(np.expand_dims(y, axis=-1)).astype(h.dtype)),axis=-1))
	ax4.set_title(' y selection', fontsize=20)
	ax4.axis('off')
	ax5.imshow( np.concatenate((h,(np.expand_dims(gt,axis=-1)).astype(h.dtype)),axis=-1))
	ax5.set_title('gt selection', fontsize=20)
	ax5.axis('off')
	ax6.imshow( h )
	ax6.set_title('Net input', fontsize=20)
	ax6.axis('off')
	fig1.savefig(filename)
	fig1.close()

def segment_image(image, model, block_size=128, overlap_fraction=2/3):

	# Calculate overlap in pixels
	overlap = int(block_size * overlap_fraction)
	step = block_size - overlap

	# Image dimensions
	height, width = image.shape[:2]

	# Padding image to handle edges
	pad_height = (height % step) if (height % step) != 0 else step
	pad_width = (width % step) if (width % step) != 0 else step
	padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

	# Dimensions of padded image
	padded_height, padded_width = padded_image.shape[:2]

	# Initialize the array to hold the segmented image
	segmented_image = np.zeros((padded_height, padded_width))

	# Create an array to count the votes for each pixel
	vote_count = np.zeros((padded_height, padded_width))

	# Segment the image in blocks
	i = 0
	for y in range(0, padded_height - block_size + 1, step):
		'''
		for x in range(0, padded_width - block_size + 1, step):
			# Extract the block
			block = padded_image[y:y + block_size, x:x + block_size]
			block = np.expand_dims(block, axis=0)  # Add batch dimension
			# Perform segmentation
			prediction = model.predict(block/255, verbose=0)
			prediction = np.squeeze(prediction)  # Remove batch dimension

			# Add votes to the segmented image
			segmented_image[y:y + block_size, x:x + block_size] += prediction[:,:,1]
			vote_count[y:y + block_size, x:x + block_size] += 1
		'''
		blocks = []
		for x in range(0, padded_width - block_size + 1, step):
			# Extract the block
			block = padded_image[y:y + block_size, x:x + block_size]
			block = np.expand_dims(block, axis=0)  # Add batch dimension
			blocks.append(block/255)
		# Perform segmentation
		prediction = model.predict(np.concatenate(blocks),verbose=0)
		for x in range(0, padded_width - block_size + 1, step):
			pred = np.squeeze(prediction[x//step,:,:,1])  # get one tile without batch dim
			# Add votes to the segmented image
			segmented_image[y:y + block_size, x:x + block_size] += pred
			vote_count[y:y + block_size, x:x + block_size] += 1
		if i < int(100*y/(padded_height - block_size + 1))
			i = int(100*y/(padded_height - block_size + 1))
			print(f'{i}/100')

	# Calculate the final segmentation by majority voting
	vote_count[vote_count == 0] = 1
	final_segmentation = (segmented_image / vote_count) > 0.5

	# Crop to original image size
	final_segmentation = final_segmentation[:height, :width]

	return np.uint8(final_segmentation)

if __name__ == '__main__':
	model = tf.keras.models.load_model('../trained/best_model_gm3.keras')
	rgb = load_tiff_image('../others/grajau/286D2_rgb.tif')
	nir = load_tiff_image('../others/grajau/286D2_cir.tif')[:,:,:1]
	header = gdal.Open('../others/grajau/286D2_rgb.tif')
	projection = header.GetProjection()
	geotransform = header.GetGeoTransform()
	h,w = min((rgb.shape[0],nir.shape[0])), min((rgb.shape[1],nir.shape[1])) 
	img = np.concatenate([rgb[:h,:w,:],nir[:h,:w,:]],axis=-1)
	#img = np.concatenate([rgb,nir],axis=-1)
	del nir
	segmented_image = segment_image(img, model)*127 + 128
	result_image = np.concatenate([rgb,np.expand_dims(segmented_image, axis=-1)], axis=-1)
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create("../segmented.tif", img.shape[1], img.shape[0], 4, gdal.GDT_Byte)
	dataset.SetProjection(projection)
	dataset.SetGeoTransform(geotransform)
	for i in range(4):
		ds_band = dataset.GetRasterBand(i + 1)
		ds_band.WriteArray(result_image[:,:,i])
	dataset,header = None,None
	
	#plt.figure(figsize=(w//100, h//100))
	#plt.imshow(np.concatenate([rgb,np.expand_dims(segmented_image, axis=-1)], axis=-1))
	#plt.axis('off')
	#plt.savefig("../segmented2.png")
	'''
	for k in range(570):
		i = k + 1000#+random.randint(0,500)
		pred = predict_image(model, f'../data/test_imgs/chip_{i}.npy', 128)
		if pred:
			y,h = pred
		else:
			continue
		plot_pred(f'../data/test_grdt/chip_{i}.npy', y, h, 
		          f"../data/predict_vs_gt_tested/prediction_{i}.png")
	'''

