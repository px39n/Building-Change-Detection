import matplotlib.pylab as plt
from glob import *

def show_img(ds):
#show  3 group images
    grid=10
    num=4
    plt.figure(figsize=(4*grid,3*grid))
    for img,label in ds.take(1):
        for i in range(num):
            plt.subplot(num, 3, i*3+1)
            plt.axis('off')
            plt.imshow(img.numpy()[i][:,:,0:3])
            plt.subplot(num, 3, i*3+2)
            plt.axis('off')
            plt.imshow(img.numpy()[i][:,:,3:6])
            plt.subplot(num, 3, i*3+3)
            plt.axis('off')
            plt.imshow(label.numpy()[i])
    plt.show()
