import os
import numpy as np
from PIL import Image
import tensorflow as tf
from glob import *
import matplotlib.pylab as plt
from utils import *

def parse_img(img,label):
	img= tf.cast(img, tf.float32)
	img1= tf.image.random_brightness(img[0], max_delta=0.5)
	img2= tf.image.random_brightness(img[1], max_delta=0.5)
	img=tf.concat([img1,img2],axis=-1)
	img_label=tf.concat([img, label], axis=-1)
	img = (img / 127.5) - 1
	img_label = tf.image.random_crop(img_label, size=[256,256,7])
	img_label=tf.image.random_flip_left_right(img_label)
	return img_label[:,:,0:6], img_label[:,:,6,tf.newaxis]

def gen(path):
	path=path.decode("utf-8")

	ds_dir=path+"A\\"
	ds_dir2=path+"B\\"
	label_dir=path+"label\\"
	list_dir = os.listdir(ds_dir)
	for i in range(len(list_dir)):
		img1 = np.array(Image.open(ds_dir+list_dir[i]))
		img2=np.array(Image.open(ds_dir2+list_dir[i]))
		ds=[img1,img2]
		label=np.array(Image.open(label_dir+list_dir[i]))[:,:,np.newaxis]
		ds,label=parse_img(ds,label)

		yield (ds,label)
