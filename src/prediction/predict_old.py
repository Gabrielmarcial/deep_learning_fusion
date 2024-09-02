import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.models.resunet import ResUNet
from src.data.preprocess import *

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

if __name__ == '__main__':
	model = tf.keras.models.load_model('../trained/best_model_gm3.keras')
	for k in range(570):
		i = k + 1000#+random.randint(0,500)
		pred = predict_image(model, f'../data/test_imgs/chip_{i}.npy', 128)
		if pred:
			y,h = pred
		else:
			continue
		plot_pred(f'../data/test_grdt/chip_{i}.npy', y, h, 
		          f"../data/predict_vs_gt_tested/prediction_{i}.png")
