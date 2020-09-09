#Imports
from flask import Flask, render_template, request, redirect, url_for, jsonify
import shutil
from uuid import uuid4
from keras.layers.convolutional import Conv2D, AtrousConvolution2D
from keras.layers import Activation, Dense, Input, Conv2DTranspose, Dense, Flatten
from keras.layers import ReLU, Dropout, Concatenate, BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D
import keras.backend as K
import tensorflow as tf
import os
from os import getcwd
import numpy as np
import PIL
import cv2
import IPython.display
from IPython.display import clear_output
from datetime import datetime
#from dataloader import Data, TestData
try:
	from keras_contrib.layers.normalization import InstanceNormalization
except Exception:
	from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

#App Initialize
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		image = request.files['image']
		# Extracting the File extension
		nam, ext = image.filename.split('.')
		# Constructing Unique file name
		name = str(uuid4()) + f'.{ext}'
		save_path = f'{getcwd()}/static/images/{name}'
		# Saving the file
		image.save(save_path)
		# Parsing the file
		resp = predict(name)
		# Returning the response
		# return jsonify(resp)
		return render_template('result.html', p_img = name)
	else:
		return redirect(url_for('index'))


@app.route('/about')
def about():
	return render_template('about.html')

################################################################################################################################################
# Saves Model in every N minutes
TIME_INTERVALS = 2
SHOW_SUMMARY = True
INPUT_SHAPE = (256, 256, 3)
EPOCHS = 500
BATCH = 1
# 25% i.e 64 width size will be mask from both side
MASK_PERCENTAGE = .25
EPSILON = 1e-9
ALPHA = 0.0004
CHECKPOINT = "checkpoint/"
SAVED_IMAGES = "saved_images/"

#Discriminator

def dcrm_loss(y_true, y_pred):
	return -tf.reduce_mean(tf.log(tf.maximum(y_true, EPSILON)) + tf.log(tf.maximum(1. - y_pred, EPSILON)))

d_input_shape = (INPUT_SHAPE[0], int(INPUT_SHAPE[1] * (MASK_PERCENTAGE *2)), INPUT_SHAPE[2])
d_dropout = 0.25
DCRM_OPTIMIZER = Adam(0.0001, 0.5)

def d_build_conv(layer_input, filter_size, kernel_size=4, strides=2, activation='leakyrelu', dropout_rate=d_dropout, norm=True):
	c = Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
	if activation == 'leakyrelu':
		c = LeakyReLU(alpha=0.2)(c)
	if dropout_rate:
		c = Dropout(dropout_rate)(c)
	if norm == 'inst':
		c = InstanceNormalization()(c)
	return c


def build_discriminator():
	d_input = Input(shape=d_input_shape)
	d = d_build_conv(d_input, 32, 5,strides=2, norm=False)

	d = d_build_conv(d, 64, 5, strides=2)
	d = d_build_conv(d, 64, 5, strides=2)
	d = d_build_conv(d, 128, 5, strides=2)
	d = d_build_conv(d, 128, 5, strides=2)
	
	flat = Flatten()(d)
	fc1 = Dense(1024, activation='relu')(flat)
	d_output = Dense(1, activation='sigmoid')(fc1)
	
	return Model(d_input, d_output)


# Discriminator initialization
DCRM = build_discriminator()
DCRM.compile(loss=dcrm_loss, optimizer=DCRM_OPTIMIZER)

#Generator Model

def gen_loss(y_true, y_pred):
	G_MSE_loss = K.mean(K.square(y_pred - y_true))
	return G_MSE_loss - ALPHA * tf.reduce_mean(tf.log(tf.maximum(y_pred, EPSILON)))

g_input_shape = (INPUT_SHAPE[0], int(INPUT_SHAPE[1] * (MASK_PERCENTAGE *2)), INPUT_SHAPE[2])
g_dropout = 0.25
GEN_OPTIMIZER = Adam(0.001, 0.5)

def g_build_conv(layer_input, filter_size, kernel_size=4, strides=2, activation='leakyrelu', dropout_rate=g_dropout, norm='inst', dilation=1):
	c = AtrousConvolution2D(filter_size, kernel_size=kernel_size, strides=strides,atrous_rate=(dilation,dilation), padding='same')(layer_input)
	if activation == 'leakyrelu':
		c = ReLU()(c)
	if dropout_rate:
		c = Dropout(dropout_rate)(c)
	if norm == 'inst':
		c = InstanceNormalization()(c)
	return c


def g_build_deconv(layer_input, filter_size, kernel_size=3, strides=2, activation='relu', dropout=0):
	d = Conv2DTranspose(filter_size, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
	if activation == 'relu':
		d = ReLU()(d)
	return d


def build_generator():
	g_input = Input(shape=g_input_shape)
	
	g1 = g_build_conv(g_input, 64, 5, strides=1)
	g2 = g_build_conv(g1, 128, 4, strides=2)
	g3 = g_build_conv(g2, 256, 4, strides=2)

	g4 = g_build_conv(g3, 512, 4, strides=1)
	g5 = g_build_conv(g4, 512, 4, strides=1)
	
	g6 = g_build_conv(g5, 512, 4, strides=1, dilation=2)
	g7 = g_build_conv(g6, 512, 4, strides=1, dilation=4)
	g8 = g_build_conv(g7, 512, 4, strides=1, dilation=8)
	g9 = g_build_conv(g8, 512, 4, strides=1, dilation=16)
	
	g10 = g_build_conv(g9, 512, 4, strides=1)
	g11 = g_build_conv(g10, 512, 4, strides=1)
	
	g12 = g_build_deconv(g11, 256, 4, strides=2)
	g13 = g_build_deconv(g12, 128, 4, strides=2)
	
	g14 = g_build_conv(g13, 128, 4, strides=1)
	g15 = g_build_conv(g14, 64, 4, strides=1)
	
	g_output = AtrousConvolution2D(3, kernel_size=4, strides=(1,1), activation='tanh',padding='same', atrous_rate=(1,1))(g15)
	
	return Model(g_input, g_output)


# Generator Initialization
GEN = build_generator()
GEN.compile(loss=gen_loss, optimizer=GEN_OPTIMIZER)

#Combined Model

IMAGE = Input(shape=g_input_shape)
DCRM.trainable = False
GENERATED_IMAGE = GEN(IMAGE)
CONF_GENERATED_IMAGE = DCRM(GENERATED_IMAGE)

COMBINED = Model(IMAGE, [CONF_GENERATED_IMAGE, GENERATED_IMAGE])
COMBINED.compile(loss=['mse', 'mse'], optimizer=GEN_OPTIMIZER)

#Masking and Demasking

def mask_width(img):
	image = img.copy()
	height = image.shape[0]
	width = image.shape[1]
	new_width = int(width * MASK_PERCENTAGE)
	mask = np.ones([height, new_width, 3])
	missing_x = img[:, :new_width]
	missing_y = img[:, width - new_width:]
	missing_part = np.concatenate((missing_x, missing_y), axis=1)
	image = image[:, :width - new_width]
	image = image[:, new_width:]
	return image, missing_part


def get_masked_images(images):
	mask_images = []
	missing_images = []
	for image in images:
		mask_image, missing_image = mask_width(image)
		mask_images.append(mask_image)
		missing_images.append(missing_image)
	return np.array(mask_images), np.array(missing_images)


def get_demask_images(original_images, generated_images):
	demask_images = []
	for o_image, g_image in zip(original_images, generated_images):
		width = g_image.shape[1] // 2
		x_image = g_image[:, :width]
		y_image = g_image[:, width:]
		o_image = np.concatenate((x_image,o_image, y_image), axis=1)
		demask_images.append(o_image)
	return np.asarray(demask_images)


def save_model():
	global DCRM, GEN
	models = [DCRM, GEN]
	model_names = ['DCRM','GEN']

	for model, model_name in zip(models, model_names):
		model_path =  CHECKPOINT + "%s.json" % model_name
		weights_path = CHECKPOINT + "/%s.hdf5" % model_name
		options = {"file_arch": model_path, 
					"file_weight": weights_path}
		json_string = model.to_json()
		open(options['file_arch'], 'w').write(json_string)
		model.save_weights(options['file_weight'])
	print("Saved Model")
	
	
def load_model():
	# Checking if all the model exists
	model_names = ['DCRM', 'GEN']
	files = os.listdir(CHECKPOINT)
	for model_name in model_names:
		if model_name+".json" not in files or\
		   model_name+".hdf5" not in files:
			print("Models not Found")
			return
	global DCRM, GEN, COMBINED, IMAGE, GENERATED_IMAGE, CONF_GENERATED_IMAGE
	
	# load DCRM Model
	model_path = CHECKPOINT + "%s.json" % 'DCRM'
	weight_path = CHECKPOINT + "%s.hdf5" % 'DCRM'
	with open(model_path, 'r') as f:
		DCRM = model_from_json(f.read())
	DCRM.load_weights(weight_path)
	DCRM.compile(loss=dcrm_loss, optimizer=DCRM_OPTIMIZER)
	
	#load GEN Model
	model_path = CHECKPOINT + "%s.json" % 'GEN'
	weight_path = CHECKPOINT + "%s.hdf5" % 'GEN'
	with open(model_path, 'r') as f:
		 GEN = model_from_json(f.read(), custom_objects={'InstanceNormalization': InstanceNormalization()})
	GEN.load_weights(weight_path)
	
	# Combined Model
	DCRM.trainable = False
	IMAGE = Input(shape=g_input_shape)
	GENERATED_IMAGE = GEN(IMAGE)
	CONF_GENERATED_IMAGE = DCRM(GENERATED_IMAGE)

	COMBINED = Model(IMAGE, [CONF_GENERATED_IMAGE, GENERATED_IMAGE])
	COMBINED.compile(loss=['mse', 'mse'], optimizer=GEN_OPTIMIZER)
	
	print("loaded model")
	
	
def save_image(epoch, steps):
	train_image = test_data.get_data(1)
	if train_image is None:
		train_image = test_data.get_data(1)
		
	test_image = data.get_data(1)
	if test_image is None:
		test_image = test_data.get_data(1)
	
	for nc, original in enumerate([train_image, test_image]):
		if nc:
			print("Predicting with train image")
		else:
			print("Predicting with test image")
			
		mask_image_original , missing_image = get_masked_images(original)
		mask_image = mask_image_original.copy()
		mask_image = mask_image / 127.5 - 1
		missing_image = missing_image / 127.5 - 1
		gen_missing = GEN.predict(mask_image)
		gen_missing = (gen_missing + 1) * 127.5
		gen_missing = gen_missing.astype(np.uint8)
		demask_image = get_demask_images(mask_image_original, gen_missing)

		mask_image = (mask_image + 1) * 127.5
		mask_image = mask_image.astype(np.uint8)

		border = np.ones([original[0].shape[0], 10, 3]).astype(np.uint8)

		file_name = str(epoch) + "_" + str(steps) + ".jpg"
		final_image = np.concatenate((border, original[0],border,mask_image_original[0],border, demask_image[0], border), axis=1)
		if not nc:
			cv2.imwrite(os.path.join(SAVED_IMAGES, file_name), final_image)
		final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)


def save_log(log):
	with open('log.txt', 'a') as f:
		f.write("%s\n"%log)


#Recursive Paint

def recursive_paint(image, factor=3):
	final_image = None
	gen_missing = None
	for i in range(factor):
		demask_image = None
		if i == 0:
			x, y = get_masked_images([image])
			gen_missing = GEN.predict(x)
			final_image = get_demask_images(x, gen_missing)[0]
		else:
			gen_missing = GEN.predict(gen_missing)
			final_image = get_demask_images([final_image], gen_missing)[0]
	return final_image

load_model()
graph = tf.get_default_graph()


def predict(file_name):
	global graph
	with graph.as_default():
		save_dir = f'{getcwd()}/static/images/'
		image = cv2.imread(save_dir + file_name)
		image = cv2.resize(image, (256,256))
		cropped_image = image[:, 65:193]
		input_image = cropped_image / 127.5 - 1
		input_image = np.expand_dims(input_image, axis=0)
		print(input_image.shape)
		predicted_image = GEN.predict(input_image)
		predicted_image = get_demask_images(input_image, predicted_image)[0]
		predicted_image = (predicted_image + 1) * 127.5
		predicted_image = predicted_image.astype(np.uint8)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		cv2.imwrite(os.path.join(f'{getcwd()}/static/processed/', file_name), predicted_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
	return True
############################################################################################################################################    

if __name__ == '__main__':
	app.run(debug=True)
