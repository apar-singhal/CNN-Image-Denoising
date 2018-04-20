# # Assignment 3

# #### Importing
import numpy as np
import os
import cv2
#import scipy
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Conv2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam


#Random Seed
np.random.seed(123456)

# ### Paths to training, validating and testing data (Berkeley Segmentation Dataset)
path_folder = ".\BSR"
path_train_img = os.path.join(path_folder, "train")
path_val_img = os.path.join(path_folder, "val")
path_test_img = os.path.join(path_folder, "test")

### Resizing images
def grayscale_resize_images(path_images, fraction):
    path_save_images = path_images + os.path.sep + 'resized' + os.path.sep + 'clean'
    for filename in os.listdir(path_images):
        image_gray = scipy.misc.imread(path_images + os.path.sep + filename, True)
        image_gray_resized = scipy.misc.imresize(arr=image_gray, size=fraction, interp='bicubic')
        scipy.misc.imsave(path_save_images + os.path.sep + filename, image_gray_resized)


fraction = 0.5
grayscale_resize_images(path_train_img, fraction)
print ('Train clean resized images saved')
grayscale_resize_images(path_val_img, fraction)
print ('Validation clean resized images saved')
grayscale_resize_images(path_test_img, fraction)
print ('Test clean resized images saved')


### Adding noise to resized images
def add_gaussian_noise(path_images, mean, sigma):
    path_save_images = path_images + os.path.sep + '..' + os.path.sep + 'noisy'
    for filename in os.listdir(path_images):
        image_clean = scipy.misc.imread(path_images + os.path.sep + filename)
        noise = np.random.normal(mean, sigma, image_clean.shape)
        image_noisy = image_clean + noise
        scipy.misc.imsave(path_save_images + os.path.sep + filename, image_noisy)


mean, sigma = 0, 25
path_clean = 'resized' + os.path.sep + 'clean'
add_gaussian_noise(path_train_img + os.path.sep + path_clean, mean, sigma)
print ('Train noisy resized images saved')
add_gaussian_noise(path_val_img + os.path.sep + path_clean, mean, sigma)
print ('Validation noisy resized images saved')
add_gaussian_noise(path_test_img + os.path.sep + path_clean, mean, sigma)
print ('Test noisy resized images saved')

## IN FOLDER_STRUCTURE PROVIDE PATH TO BSR FOLDER AS PER YOUR SYSTEM
path_train_img = os.path.join(path_folder, "train")
path_train_clean = os.path.join(path_folder, "train/resized/clean")
path_val_clean = os.path.join(path_folder, "val/resized/clean")
path_test_clean = os.path.join(path_folder, "test/resized/clean")

path_train_noisy = os.path.join(path_folder, "train/resized/noisy")
path_val_noisy = os.path.join(path_folder, "val/resized/noisy")
path_test_noisy = os.path.join(path_folder, "test/resized/noisy")


# ## Patches

# Our images are of size 160X240 and we want to create patches of 50X50
# 
# With step size of 1X1, we will have 21201 patches ((160-49)X(240-49))
# Since we need around 400,000 patches from 200 images, we need around 2000 patches per image
# 
# As we saw, step size of 1X1 gave 21,201 patches per image
# Similarly, step size of 3X3 will give 2,368 patches per image
# And step size of 4X4 will give 1,344 patches per image
# 
# I am taking stride of 3X3 and will have total of 473,600 patches each from training and testing set and 236,800 patches from validation set.

# ### Method for Sliding window
def sliding_window(image, window_shape, step_size):
    rows = image.shape[0]
    cols = image.shape[1]
    for x in range(0, rows, step_size):
        for y in range(0, cols, step_size):
            yield(x, y, image[x:x+window_shape[0], y:y+window_shape[1]])


# ### Method for Creating clean patches
def create_patches(path_images, window_shape, step_size, max_count):
    patches = []
    print ('Creating ' + str(window_shape[0]) + 'X' + str(window_shape[1]) + ' patches for images in ' + path_images)
    for i, filename in enumerate(os.listdir(path_images)):
        image = cv2.imread(path_images + os.path.sep + filename, cv2.IMREAD_GRAYSCALE)
        for (x, y, window) in sliding_window(image, window_shape, step_size):
            if(window.shape == window_shape):
                patches.append(window)
                
    print ('All Patched Created added to list. Removing extra patches (if any)')
    while len(patches) > max_count:
        patches.pop(np.random.randint(0, len(patches)))
    print('Done')
    return np.array(patches)


# ### Patch Sizes
img_x = 50
img_y = 50
window_shape = (img_x, img_y)
step_size = 3
max_count = 400000


# ### Clean Patches
clean_patches_train = create_patches(path_train_clean, window_shape, step_size, max_count)
clean_patches_val = create_patches(path_val_clean, window_shape, step_size, max_count/2)
clean_patches_test = create_patches(path_test_clean, window_shape, step_size, max_count)


# #### Validating
print('Before Preprocessing')
print ('Clean Train Patches Shape:', clean_patches_train.shape)
print ('Clean Validation Patches Shape:', clean_patches_val.shape)
print ('Clean Test Patches Shape:', clean_patches_test.shape)

# plt.imshow(clean_patches_train[0])
# plt.show()


# #### Reshaping
clean_patches_train = clean_patches_train.reshape(clean_patches_train.shape[0], img_x, img_y, 1)
clean_patches_val = clean_patches_val.reshape(clean_patches_val.shape[0], img_x, img_y, 1)
clean_patches_test = clean_patches_test.reshape(clean_patches_test.shape[0], img_x, img_y, 1)


# #### Normalizing Patches
clean_patches_train = clean_patches_train.astype('float32')
clean_patches_val = clean_patches_val.astype('float32')
clean_patches_test = clean_patches_test.astype('float32')

clean_patches_train /= 255.
clean_patches_val /= 255.
clean_patches_test /= 255.


# #### Validating
print('After Pre-processing')
print ('Clean Train Patches Shape:', clean_patches_train.shape)
print ('Clean Validation Patches Shape:', clean_patches_val.shape)
print ('Clean Test Patches Shape:', clean_patches_test.shape)

# plt.imshow(clean_patches_train[0][:,:,0])
# plt.show()


# ### Noisy Patches
noisy_patches_train = create_patches(path_train_noisy, window_shape, step_size, max_count)
noisy_patches_val = create_patches(path_val_noisy, window_shape, step_size, max_count/2)
noisy_patches_test = create_patches(path_test_noisy, window_shape, step_size, max_count)


# #### Validating
print('Before Pre-processing')
print ('Noisy Train Patches Shape:', noisy_patches_train.shape)
print ('Noisy Validation Patches Shape:', noisy_patches_val.shape)
print ('Noisy Test Patches Shape:', noisy_patches_test.shape)

# plt.imshow(noisy_patches_train[0])
# plt.show()


# #### Reshaping
noisy_patches_train = noisy_patches_train.reshape(noisy_patches_train.shape[0], img_x, img_y, 1)
noisy_patches_val = noisy_patches_val.reshape(noisy_patches_val.shape[0], img_x, img_y, 1)
noisy_patches_test = noisy_patches_test.reshape(noisy_patches_test.shape[0], img_x, img_y, 1)


# #### Normalizing
noisy_patches_train = noisy_patches_train.astype('float32')
noisy_patches_val = noisy_patches_val.astype('float32')
noisy_patches_test = noisy_patches_test.astype('float32')

noisy_patches_train /= 255.
noisy_patches_val /= 255.
noisy_patches_test /= 255.


# #### Validating
print('After Pre-processing')
print ('Noisy Train Patches Shape:', noisy_patches_train.shape)
print ('Noisy Validation Patches Shape:', noisy_patches_val.shape)
print ('Noisy Test Patches Shape:', noisy_patches_test.shape)

# plt.imshow(noisy_patches_train[0][:,:,0])
# plt.show()


# ### Residual Noise
residual_patches_train = noisy_patches_train - clean_patches_train
residual_patches_val = noisy_patches_val - clean_patches_val
residual_patches_test = noisy_patches_test - clean_patches_test

print ('Residual Train Patches Shape:', residual_patches_train.shape)
print ('Residual Validation Patches Shape:', residual_patches_val.shape)
print ('Residual Test Patches Shape:', residual_patches_test.shape)

# plt.subplot(121)
# plt.imshow(residual_patches_train[0][:,:,0])
# plt.subplot(122)
# plt.imshow((noisy_patches_train[0] - clean_patches_train[0])[:,:,0])
# plt.show()


# # Models

# ### Common info
input_shape = (img_x, img_y, 1)
kernel_size = (3,3)
depth = 17
batch_size = 128
epoch = 50

# ### Optimizers
sgd = SGD(momentum=0.9, decay=.001)
adam = Adam()


# ## Model with Batch Normalization
model_bn = Sequential()
model_bn.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape))
for i in range(depth-2):
    model_bn.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same'))
    model_bn.add(BatchNormalization())
    model_bn.add(Activation('relu'))
model_bn.add(Conv2D(filters=1, kernel_size=kernel_size, padding='same', input_shape=input_shape))
# print(len(model_bn.layers))
# print(model_bn.output_shape)


# ### SGD with BatchNormalization

# #### Input: Noisy image; Output: Residual noise
model_bn.compile(loss='mean_squared_error', optimizer=sgd)

history_bn_sgd_residual = model_bn.fit(noisy_patches_train, residual_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, residual_patches_val)
         )


plt.plot(history_bn_sgd_residual.history['loss'])
plt.plot(history_bn_sgd_residual.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_bn_sgd_residual.jpg")
# plt.show()

# output_bn_sgd_residual = model_bn.predict(noisy_patches_test)
# output_bn_sgd_residual.shape

model_bn.save_weights("weights_bn_sgd_residual.h5")
print ('Weights Saved : weights_bn_sgd_residual')


# #### Input: Noisy image; Output: Denoised image
model_bn.compile(loss='mean_squared_error', optimizer=sgd)

history_bn_sgd_image = model_bn.fit(noisy_patches_train, clean_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, clean_patches_val)
         )

plt.plot(history_bn_sgd_image.history['loss'])
plt.plot(history_bn_sgd_image.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_bn_sgd_image.jpg")
# plt.show()

# output_bn_sgd_image = model_bn.predict(noisy_patches_test)
# output_bn_sgd_image.shape

model_bn.save_weights("weights_bn_sgd_image.h5")
print ('weights_bn_sgd_image Saved')


# ### ADAM with BatchNormalization

# #### Input: Noisy image; Output: Residual noise
model_bn.compile(loss='mean_squared_error', optimizer=adam)

history_bn_adam_residual = model_bn.fit(noisy_patches_train, residual_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, residual_patches_val)
         )

plt.plot(history_bn_adam_residual.history['loss'])
plt.plot(history_bn_adam_residual.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_bn_adam_residual.jpg")
# plt.show()

# output_bn_adam_residual = model_bn.predict(noisy_patches_test)
# output_bn_adam_residual.shape

model_bn.save_weights("weights_bn_adam_residual.h5")
print ('weights_bn_adam_residual Saved')


# #### Input: Noisy image; Output: Denoised image
model_bn.compile(loss='mean_squared_error', optimizer=adam)

history_bn_adam_image = model_bn.fit(noisy_patches_train, clean_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, clean_patches_val)
         )

plt.plot(history_bn_adam_image.history['loss'])
plt.plot(history_bn_adam_image.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_bn_adam_image.jpg")
# plt.show()

# output_bn_adam_image = model_bn.predict(noisy_patches_test)
# output_bn_adam_image.shape

model_bn.save_weights("weights_bn_adam_image.h5")
print ('weights_bn_adam_image Saved')


# ## Model Without Batch normalization
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape))
for i in range(depth-2):
    model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='same'))
    model.add(Activation('relu'))
model.add(Conv2D(filters=1, kernel_size=kernel_size, padding='same', input_shape=input_shape))
# print(len(model.layers))
# print(model.output_shape)


# ### SGD without BatchNormalization

# #### Input: Noisy image; Output: Residual noise
model.compile(loss='mean_squared_error', optimizer=sgd)

history_sgd_residual = model.fit(noisy_patches_train, residual_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, residual_patches_val)
         )

plt.plot(history_sgd_residual.history['loss'])
plt.plot(history_sgd_residual.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_sgd_residual.jpg")
# plt.show()

# output_sgd_residual = model.predict(noisy_patches_test)
# output_sgd_residual.shape

model_bn.save_weights("weights_sgd_residual.h5")
print ('weights_sgd_residual saved!')


# #### Input: Noisy image; Output: Denoised image
model.compile(loss='mean_squared_error', optimizer=sgd)
history_sgd_image = model.fit(noisy_patches_train, clean_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, clean_patches_val)
         )

plt.plot(history_sgd_image.history['loss'])
plt.plot(history_sgd_image.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_sgd_image.jpg")
# plt.show()

# output_sgd_image = model.predict(noisy_patches_test)
# output_sgd_image.shape

model_bn.save_weights("weights_sgd_image.h5")
print('weights_sgd_image saved')


# ### ADAM without BatchNormalization

# #### Input: Noisy image; Output: Residual noise
model.compile(loss='mean_squared_error', optimizer=adam)

history_adam_residual = model.fit(noisy_patches_train, residual_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, residual_patches_val)
         )

plt.plot(history_adam_residual.history['loss'])
plt.plot(history_adam_residual.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_adam_residual.jpg")
# plt.show()

# output_adam_residual = model.predict(noisy_patches_test)
# output_adam_residual.shape

model_bn.save_weights("weights_adam_residual.h5")
print('weights_adam_residual saved')


# #### Input: Noisy image; Output: Denoised image
model.compile(loss='mean_squared_error', optimizer=adam)

history_adam_image = model.fit(noisy_patches_train, clean_patches_train,
             batch_size = batch_size,
             epochs=epoch,
             validation_data = (noisy_patches_val, clean_patches_val)
         )

plt.plot(history_adam_image.history['loss'])
plt.plot(history_adam_image.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("history_adam_image.jpg")
# plt.show()

# output_adam_image = model.predict(noisy_patches_test)
# output_adam_image.shape

model_bn.save_weights("weights_adam_image.h5")
print('weights_adam_image saved')


# ## Comparing Results

# #### Comparing SGD and Adam optimizers
plt.plot(history_bn_adam_residual.history['loss'])
plt.plot(history_bn_sgd_residual.history['loss'])
plt.title('Comparing SGD and Adam')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Adam', 'SGD'], loc='upper left')
plt.savefig("comparison_sgd_adam.jpg")
# plt.show()


# #### Comparing Residual Noise learning vs Clean Image learning

# ##### SGD
plt.plot(history_bn_sgd_residual.history['loss'])
plt.plot(history_bn_sgd_image.history['loss'])
plt.title('SGD: Learning Noise v/s Image')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Noise learning', 'Image learning'], loc='upper left')
plt.savefig("comparison_noise_image_sgd.jpg")
# plt.show()


# ##### Adam
plt.plot(history_bn_adam_residual.history['loss'])
plt.plot(history_bn_adam_image.history['loss'])
plt.title('Adam: Learning Noise v/s Image')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Noise learning', 'Image learning'], loc='upper left')
plt.savefig("comparison_noise_image_adam.jpg")
# plt.show()


# ### PSNR
MAX = 1 #We normalised the data
def calculate_psnr(mse):
    return 20*math.log10(MAX) - 10*math.log10(mse)

def list_psnr(list_mse):
    list_psnr= []
    for mse in list_mse:
        psnr = calculate_psnr(mse)
        list_psnr.append(psnr)
    return list_psnr


# #### Comparing With and Without Batch Normalization

# ##### SGD
psnr_with_BN_sgd = list_psnr(history_bn_sgd_residual.history['loss'])
psnr_without_BN_sgd = list_psnr(history_sgd_residual.history['loss'])

plt.plot(psnr_with_BN_sgd)
plt.plot(psnr_without_BN_sgd)
plt.title('SGD: With and Without BN')
plt.ylabel('psnr')
plt.xlabel('patches')
plt.legend(['with BN', 'without BN'], loc='upper left')
plt.savefig("comparison_bn_sgd.jpg")
# plt.show()


# ##### Adam
psnr_with_BN_adam = list_psnr(history_bn_adam_residual.history['loss'])
psnr_without_BN_adam = list_psnr(history_adam_residual.history['loss'])

plt.plot(psnr_with_BN_adam)
plt.plot(psnr_without_BN_adam)
plt.title('Adam: With and Without BN')
plt.ylabel('psnr')
plt.xlabel('patches')
plt.legend(['with BN', 'without BN'], loc='upper left')
plt.savefig("comparison_bn_adam.jpg")
# plt.show()
