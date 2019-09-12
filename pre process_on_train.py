#%%
import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc
import os
import imageio
from scipy import misc
from keras.preprocessing import image
import scipy.stats

#%%
directory="E:/face_siamese/Towsan_final/train/"
directory2="E:/face_siamese/Towsan_final/train1/"
files = sorted(os.listdir(directory2))

test2="E:/face_siamese/Towsan_final/test1/"
test="E:/face_siamese/Towsan_final/test/"
files2 = sorted(os.listdir(test2))






#%%

for i in files:
    address=i
    anchor_img = image.load_img(directory2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)


    anchor_img = color.rgb2gray(anchor_img)
    blurred_face = ndimage.gaussian_filter(anchor_img, sigma=3)
    very_blurred = ndimage.gaussian_filter(anchor_img, sigma=5)

    local_mean = ndimage.uniform_filter(anchor_img, size=11)
    alpha = 3

    med=ndimage.median_filter(anchor_img, 2)
    bm1 = ndimage.gaussian_filter(med, sigma=1)
    bm2 = ndimage.gaussian_filter(med, sigma=4)
    sharpened = bm2 + alpha * (bm1 - med)
    x = anchor_img.mean()

    plt.imsave(directory+address, sharpened, cmap=plt.cm.gray)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    #anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    #anchor_img = image.load_img(directory+address, target_size=(224, 224,3))

    #anchor_img = image.img_to_array(anchor_img)

    #anchor_img = anchor_img / 255

    #print(np.mean(anchor_img))




#%%
for i in files2:
    address=i
    anchor_img = image.load_img(test2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    #anchor_img=scipy.stats.scoreatpercentile(anchor_img,50)

    anchor_img = color.rgb2gray(anchor_img)
    blurred_face = ndimage.gaussian_filter(anchor_img, sigma=3)
    very_blurred = ndimage.gaussian_filter(anchor_img, sigma=5)

    local_mean = ndimage.uniform_filter(anchor_img, size=11)
    alpha = 3

    med=ndimage.median_filter(anchor_img, 2)
    bm1 = ndimage.gaussian_filter(med, sigma=1)
    bm2 = ndimage.gaussian_filter(med, sigma=4)
    sharpened = bm1 + alpha * (bm1 - med)
    x = anchor_img.mean()

    plt.imsave(test+address, sharpened, cmap=plt.cm.gray)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    #anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    #anchor_img = image.load_img(test+address, target_size=(224, 224,3))

    #anchor_img = image.img_to_array(anchor_img)

    #anchor_img = anchor_img / 255

    #print(np.mean(anchor_img))



''''''

''''''

#%%

directory="E:/face_siamese/Towsan_final/train/"
directory2="E:/face_siamese/Towsan_final/train1/"
files = sorted(os.listdir(directory2))

test2="E:/face_siamese/Towsan_final/test1/"
test="E:/face_siamese/Towsan_final/test/"
files2 = sorted(os.listdir(test2))


for i in files:
    address=i
    anchor_img = image.load_img(directory2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    anchor_img = color.rgb2gray(anchor_img)
    sx = ndimage.sobel(anchor_img, axis=0, mode='nearest',cval=0.0)
    sy = ndimage.sobel(anchor_img, axis=1, mode='nearest',cval=0.0)
    sob = np.hypot(sx, sy)

    x = sob.mean()
    sob = sob/np.max(sob)
    im = 1 * sob + 1 * anchor_img
    im=im/2
    im=im-0.1
    #sob=sob+0.5
    plt.imsave(directory+address, im, cmap=plt.cm.gray, vmin=x - 0.4, vmax=x + 0.4)
    #plt.imsave(directory + address, sob, cmap=plt.cm.gray)
    '''
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(directory+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))
    '''



#%%

directory="E:/face_siamese/Towsan_final/train/"
directory2="E:/face_siamese/Towsan_final/train1/"
files = sorted(os.listdir(directory2))

test2="E:/face_siamese/Towsan_final/test1/"
test="E:/face_siamese/Towsan_final/test/"
files2 = sorted(os.listdir(test2))


for i in files2:
    address=i
    anchor_img = image.load_img(test2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    anchor_img = color.rgb2gray(anchor_img)
    mean_img_gray = np.mean(anchor_img)
    sx = ndimage.sobel(anchor_img, axis=0, mode='nearest',cval=0.0)
    sy = ndimage.sobel(anchor_img, axis=1, mode='nearest',cval=0.0)
    sob = np.hypot(sx, sy)

    x = sob.mean()
    sob = sob/np.max(sob)

    im=1*sob+ 1*anchor_img
    im=im/2
    im=im-0.1
    #im=im+(0.5*mean_img_gray)
    #sob=sob+0.5
    plt.imsave(test+address, im, cmap=plt.cm.gray, vmin=x - 0.4, vmax=x + 0.4)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(test+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))








'''

'''
#%%

for i in files:
    address=i
    anchor_img = image.load_img(directory+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    anchor_img = color.rgb2gray(anchor_img)
    mean = 0.5
    z = np.mean(anchor_img)
    m = mean - z
    anchor_img = anchor_img + m
    anchor_img = anchor_img / np.max(anchor_img)
    x = anchor_img.mean()

    plt.imsave(directory+address, anchor_img, cmap=plt.cm.gray, vmin=x - 0.5, vmax=x + 0.5)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(directory+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))




#%%
for i in files2:
    address=i
    anchor_img = image.load_img(test+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    anchor_img = color.rgb2gray(anchor_img)
    mean = 0.5
    z = np.mean(anchor_img)
    m = mean - z
    anchor_img = anchor_img + m
    anchor_img = anchor_img / np.max(anchor_img)
    x = anchor_img.mean()

    plt.imsave(test+address, anchor_img, cmap=plt.cm.gray, vmin=x - 0.5, vmax=x + 0.5)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(test+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))


















#%%
# colorful with sobel filter

directory="E:/face_siamese/Towsan_final/train-new2/"
directory2="E:/face_siamese/Towsan_final/train-new/"
files = sorted(os.listdir(directory2))

test2="E:/face_siamese/Towsan_final/test1/"
test="E:/face_siamese/Towsan_final/test2/"
files2 = sorted(os.listdir(test2))


for i in files:
    address=i
    anchor_img = image.load_img(directory2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    gray = color.rgb2gray(anchor_img)
    sx = ndimage.sobel(gray, axis=0, mode='nearest',cval=0.0)
    sy = ndimage.sobel(gray, axis=1, mode='nearest',cval=0.0)
    sob = np.hypot(sx, sy)

    #x = anchor_img.mean()
    x=sob.mean()
    sob = sob/np.max(sob)
    sob = np.repeat(sob[..., np.newaxis], 3, -1)
    '''
    sob[np.where(anchor_img == 1)] = 1
    '''
    im = 4 * sob + 4 * anchor_img
    #im=im/2
    im=im+0.2
    im=im/np.max(im)
    #sob=sob+0.5
    plt.imsave(directory+address, im)
    #plt.imsave(directory + address, sob, cmap=plt.cm.gray)
    '''
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(directory+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))
    '''



#%%

directory="E:/face_siamese/Towsan_final/train2/"
directory2="E:/face_siamese/Towsan_final/train1/"
files = sorted(os.listdir(directory2))

test2="E:/face_siamese/Towsan_final/test10/"
test="E:/face_siamese/Towsan_final/test10-new/"
files2 = sorted(os.listdir(test2))

count=1
for i in files2:
    print(count, " of ",len(files2))
    address=i
    anchor_img = image.load_img(test2+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255
    print(np.mean(anchor_img),"aaaaa")
    gray = color.rgb2gray(anchor_img)
    mean_img_gray = np.mean(anchor_img)
    sx = ndimage.sobel(gray, axis=0, mode='nearest',cval=0.0)
    sy = ndimage.sobel(gray, axis=1, mode='nearest',cval=0.0)
    sob = np.hypot(sx, sy)

    x = sob.mean()
    sob = sob/np.max(sob)
    sob = np.repeat(sob[..., np.newaxis], 3, -1)
    im=4*sob+ 4*anchor_img
    #im=im/2
    im=im+0.2
    im=im/np.max(im)
    #im=im+(0.5*mean_img_gray)
    #sob=sob+0.5
    plt.imsave(test+address, im)
    #plt.imsave(directory + address, anchor_img, cmap=plt.cm.gray)
    anchor_img = np.repeat(anchor_img[..., np.newaxis], 3, -1)


    anchor_img = image.load_img(test+address, target_size=(224, 224,3))

    anchor_img = image.img_to_array(anchor_img)

    anchor_img = anchor_img / 255

    print(np.mean(anchor_img))
    count=count+1








'''

'''