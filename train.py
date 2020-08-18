from tensorflow.keras import datasets, layers, models
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import PIL.Image as Image
import pandas as pd
import os
from model import *
from utils import *
from preprocess import *
from glob import *
import tensorflow_addons as tfa

def compile_and_fit(model, name):
    # hyperparameter optimizer,loss, epoch
    optimizer = tf.keras.optimizers.Adam(LR)
    loss = tfa.losses.contrastive_loss
    #metrics = [ 'accuracy']  # 可以有多个  通常为'accuracy'
    model.compile(optimizer=optimizer, loss=loss)
    # model.summary()
    history = model.fit(
        ds,
        epochs=EPOCH,
        #validation_data=val,
        #callbacks=[cp_callback]
    )
    return history



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#build pipeline

#initialize Model
model=snet()
#tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

#build data pipeline

ds = tf.data.Dataset.from_generator(gen,(tf.float32,tf.float32),args=[TRAIN_PATH])
ds=ds.batch(4)
val=tf.data.Dataset.from_generator(gen,(tf.float32,tf.float32),args=(VAL_PATH))
history=compile_and_fit(model,name="snet")


# for abc in ds.take(1):
#     print(abc[0].shape)
#     print(abc[1].shape)
#     print(model(abc[0]).shape)
#     #print(tfa.losses.contrastive_loss(model(abc[0]),abc[1]))
#

