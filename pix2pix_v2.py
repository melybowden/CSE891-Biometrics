import tensorflow as tf
from tensorboardX import SummaryWriter
import random

import os
import time

#from matplotlib import pyplot as plt
import tensorboard
import datetime
import glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Preprocessed using CelebAMask Data_preprocessing
PATH = "./"
TR_IMG = "train_img/"
TR_LABEL = "train_label_color/"
# TR_LABEL = "train_label/"
TE_IMG = "test_img/"
TE_LABEL = "test_label_color/"
ZIMGS = "CelebAMask-HQ-mask-anno-short/"
IDENTITIES = "face_recognition/"
CMASK = "colored_masks/"
TRAIN_SPLIT = 24000
BUFFER_SIZE = 400
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
TRAIN_OR_TEST = "train"
LAMBDA = 100
EPOCHS = 150
STEPS = 0

def load_mask(image_file):
  image_file = image_file.decode('UTF-8')
  img_dict = np.load(image_file, allow_pickle=True)
  
  faceid = (IDENTITIES + image_file.split('/')[-1])
  faceid = np.load(faceid)['arr_0']
  
  cmask = np.load(CMASK + image_file.split('/')[-1])['cmask'] / 255
  
  n = np.random.randint(30000)
  imgd2 = np.load(ZIMGS+str(n)+'.npz', allow_pickle=True)
  mask2 = imgd2['Masks']
  cm = np.matmul(mask2, np.flip(img_dict['Colors'], axis=-1)) / (np.sum(imgd2['Masks'], axis=-1, keepdims=True) + 1e-10)
  cmask2 = cm / 255
  
  return img_dict['Masks'].astype(np.float32), img_dict['Colors'].reshape((1,1,7*3)).astype(np.float32), img_dict['Image'].astype(np.float32), faceid.reshape(1,1,128).astype(np.float32), cmask.astype(np.float32), mask2.astype(np.float32), cmask2.astype(np.float32)

def load(image_file):
  masks, colors, image, faceid, cmask, mask2, cmask2 = tf.numpy_function(load_mask, [image_file], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
 
  return tf.ensure_shape(masks, [256, 256, 7]), tf.reverse(tf.ensure_shape(image, [256, 256, 3]), [-1]), tf.reverse(tf.ensure_shape(colors, [1, 1, 7*3]), [-1]), tf.ensure_shape(faceid, [1,1,128]), tf.ensure_shape(cmask, [256, 256, 3]), tf.ensure_shape(mask2, [256, 256, 7]), tf.ensure_shape(cmask2, [256, 256, 3])

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image): # normalizing the images to [-1, 1]
  #input_image = (input_image / 127.5) - 1
  #real_image = (real_image / 127.5) - 1
  input_image = input_image / 255
  real_image = real_image / 255

  return input_image, real_image

def random_jitter(input_image, real_image):
  # resizing to 256 x 256 x 3
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  # randomly cropping to 256 x 256 x 3
  # input_image, real_image = random_crop(input_image, real_image)

  if random.randint(0, 1) == 0:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image, colors, faceid, cmask, mask2, cmask2 = load(image_file)
  #input_image, real_image = random_jitter(input_image, real_image)
  colors, real_image = normalize(colors, real_image)

  return input_image, real_image, colors, faceid, cmask, mask2, cmask2

def load_image_test(image_file):
  input_image, real_image, colors, faceid, cmask, mask2, cmask2 = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  colors, real_image = normalize(colors, real_image)

  return input_image, real_image, colors, faceid, cmask, mask2, cmask2


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,7])
  # Skin, Eyes, Mouth, Hair, Nose, Clothes, Accessories
  color_inputs = tf.keras.layers.Input(shape=[1, 1, 7*3])
  tar = tf.keras.layers.Input(shape=[256,256,3])
  faceid = tf.keras.layers.Input(shape=[1,1,128])
  mask2 = tf.keras.layers.Input(shape=[256,256,7])
  tar2 = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(32, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(64, 4), # (bs, 64, 64, 128)
    downsample(128, 4), # (bs, 32, 32, 256)
    downsample(256, 4), # (bs, 16, 16, 512)
    downsample(256, 4), # (bs, 8, 8, 512)
    downsample(256, 4), # (bs, 4, 4, 512)
    downsample(256, 4), # (bs, 2, 2, 512)
    downsample(256, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(256, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(256, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(256, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(256, 4), # (bs, 16, 16, 1024)
    upsample(128, 4), # (bs, 32, 32, 512)
    upsample(64, 4), # (bs, 64, 64, 256)
    upsample(32, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') # (bs, 256, 256, 3)
  #last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='sigmoid')

  x = inputs
  y = mask2
  # Downsampling through the model
  skips = []
  yskips = []
  x = tf.concat([x, tar], axis=-1)
  y = tf.concat([y, tar2], axis=-1)
  #x = tf.concat([x, inputs[:,:,:,0:1] * color_inputs[:,:,:,0:3], inputs[:,:,:,3:4] * color_inputs[:,:,:,9:12]], axis=-1)
  #y = tf.concat([y, mask2[:,:,:,0:1] * color_inputs[:,:,:,0:3], mask2[:,:,:,3:4] * color_inputs[:,:,:,9:12]], axis=-1)
  for down in down_stack:
    x = down(x)
    skips.append(x)
    y = down(y)
    yskips.append(y)
  x = tf.concat([x, color_inputs, faceid], axis=-1)
  skips = reversed(skips[:-1])
  y = tf.concat([y, color_inputs, faceid], axis=-1)
  yskips = reversed(yskips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  for up, yskip in zip(up_stack, yskips):
    y = up(y)
    y = tf.keras.layers.Concatenate()([y, yskip])

  x = (last(x) - 0.5) * 2
  y = (last(y) - 0.5) * 2
  #x = last(x)
  return tf.keras.Model(inputs=[inputs, color_inputs, tar, faceid, mask2, tar2], outputs=[x, y])

def generator_loss(disc_generated_output, gen_output, target, disc_gen2_output, input_mask, input_mask2):
#disc_generated_output: total_disc_loss, real_loss+generated_loss+gen2_loss, real_map_loss, gen_map_loss, gen2_map_loss
  input_mask_binary = tf.cast(tf.reduce_sum(input_mask, axis=-1, keepdims=True) >= 1, tf.float32)
  
  #gan_loss = LAMBDA * tf.reduce_mean(tf.ones_like(disc_generated_output[1]) - disc_generated_output[1])
  gan_loss = LAMBDA * loss_object(tf.ones_like(disc_generated_output[1]), disc_generated_output[1])
  # mean absolute errore (0.3) real/reconstructed img
  l1_loss = LAMBDA * tf.reduce_mean(tf.abs(target * input_mask_binary - gen_output[0]))

  #reconstruct_l1_loss = LAMBDA * tf.reduce_mean(tf.abs(disc_generated_output[0] - tf.image.resize(gen_output[0],disc_generated_output[0].shape[1:3])))
  gen_l1_loss = LAMBDA * tf.reduce_mean(tf.abs(disc_gen2_output[0] - tf.image.resize(input_mask2,[32,32]))) #diff mask

  total_gen_loss = gan_loss + l1_loss + gen_l1_loss #+ reconstruct_l1_loss

  return total_gen_loss, gan_loss, l1_loss, gen_l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  input_map = tf.keras.layers.Input(shape=[256, 256, 3], name='input_map')
  img = tf.keras.layers.Input(shape=[256, 256, 3], name='image')
  x = tf.keras.layers.concatenate([input_map, img])
  down1 = downsample(64, 3, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 3)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 3)(down2) # (bs, 32, 32, 256)
  zero_pad1 = tf.keras.layers.ZeroPadding2D(2)(down3)
  conv = tf.keras.layers.Conv2D(256, 3, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 32, 32, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  last = tf.keras.layers.Conv2D(3, 3, strides=1, kernel_initializer=initializer, activation='sigmoid')(batchnorm1) # (bs, 32, 32, 3)
  
  rs = tf.keras.layers.Conv2D(1, 3, strides=1)(last)
  rs = tf.keras.layers.Flatten()(rs)
  rs = tf.keras.layers.Dense(1, activation='sigmoid')(rs)

  return tf.keras.Model(inputs=[input_map, img], outputs=[last, rs])

def discriminator_loss(disc_real_output, disc_generated_output, disc_gen2_output, input_mask, input_mask2):
  # Class error
  #real_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.ones_like(disc_real_output[1]) - disc_real_output[1]))
  #generated_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.zeros_like(disc_generated_output[1]) - disc_generated_output[1]))
  #gen2_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.zeros_like(disc_gen2_output[1]) - disc_gen2_output[1]))
  
  real_loss = LAMBDA * loss_object(tf.ones_like(disc_real_output[1]), disc_real_output[1])
  generated_loss = LAMBDA * loss_object(tf.zeros_like(disc_generated_output[1]), disc_generated_output[1])
  gen2_loss = LAMBDA * loss_object(tf.zeros_like(disc_gen2_output[1]), disc_gen2_output[1])
 
  # Map loss
  #input_mask_binary = tf.cast(tf.reduce_sum(input_mask, axis=-1, keepdims=True) >= 1, tf.float32)
  #input_mask2_binary = tf.cast(tf.reduce_sum(input_mask2, axis=-1, keepdims=True) >= 1, tf.float32)
  real_map_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.image.resize(input_mask,[32,32]) - disc_real_output[0]))
  #gen_map_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.image.resize(input_mask,disc_generated_output[0].shape[1:3]) - disc_generated_output[0]))
  #gen2_map_loss = LAMBDA * tf.reduce_mean(tf.abs(tf.image.resize(input_mask2,disc_gen2_output[0].shape[1:3]) - disc_gen2_output[0]))
  
  total_disc_loss = real_loss + generated_loss + real_map_loss + gen2_loss#+ gen_map_loss  + gen2_map_loss

  return total_disc_loss, real_loss+generated_loss+gen2_loss, real_map_loss#, gen_map_loss, gen2_map_loss

def train_step(input_image, target, colors, faceid, cmask, mask2, cmask2):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #input_image = tf.matmul(input_image, colors)
    #print("MatMul Shape: {}".format(input_image.shape))
    gen_output = generator([input_image, colors, cmask, faceid, mask2, cmask2], training=True)
    #print("Generated shape: {}".format(gen_output.shape))
    #print()
    disc_real_output = discriminator([cmask,target], training=True)
    disc_generated_output = discriminator([cmask,gen_output[0]], training=True)
    disc_gen2_output = discriminator([cmask2,gen_output[1]], training=True)

    gen_total_loss, gan_loss, l1_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, disc_gen2_output, cmask, cmask2)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output, disc_gen2_output, cmask, cmask2)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  gen_op = generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  disc_op = discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with tf.control_dependencies([gen_op, disc_op]):
    op_train = tf.no_op()

  outputs = {
             'op_train':op_train, 
             'gen_total_loss':gen_total_loss, 
             'disc_loss':disc_loss, 
             'gen_l1_loss':gen_l1_loss, 
             'input_image':input_image, 
             'target':target, 
             'gen_output':gen_output, 
             'colors':colors, 
             'faceid':faceid, 
             'cmask':cmask, 
             'mask2':mask2,
             'l1_loss':l1_loss, 
             'cmask2':cmask2,
             'disc_real_output':disc_real_output,
             'disc_generated_output':disc_generated_output,
             'disc_gen2_output':disc_gen2_output
             }
  return outputs
  #return op_train, gen_total_loss, disc_loss, gen_l1_loss, input_image, target, gen_output, colors, faceid, cmask, mask2

def fit(op, epochs, save_path, last_epoch=0):
  spe = STEPS // BATCH_SIZE
  print('Steps per batch: {0}'.format(spe))
  for epoch in range(last_epoch, epochs):
    start = time.time()
    for s in range(spe):
      se = epoch * spe + s
      res = S.run(op)
      if se % 50 == 0:
      #disc_loss: total_disc_loss, real_loss+generated_loss+gen2_loss, real_map_loss, gen_map_loss, gen2_map_loss
        print('\rEpoch: {0}, Step: {1}, GLoss: {2:.2f}, DLoss {3:.2f}, L1 {4:.2f}, Gen2L1 {5:.2f}'.format(epoch, s, res['gen_total_loss'], res['disc_loss'][0], res['l1_loss'], res['gen_l1_loss']), end='')
        write_tfboard(res['gen_total_loss'], se, 'Gen Total Loss', 'scalar')
        write_tfboard(res['disc_loss'][0], se, 'Disc Loss', 'scalar')
        write_tfboard(res['disc_loss'][2], se, 'Disc RM Loss', 'scalar')
        write_tfboard(res['gen_l1_loss'], se, 'Gen L1 Loss', 'scalar')
        write_tfboard(res['l1_loss'], se, 'L1 Loss', 'scalar')
      if se > 0 and se % 50 == 0:
        #write_tfboard(tf.matmul(tf.constant(res[4]), tf.constant(res[7])), se, 'Input Image', 'image')
        # res[4] 256x256x7 res[7] 7x3
        #other_mask = np.stack([res['mask2'][:,:,:,0], res['mask2'][:,:,:,3], res['mask2'][:,:,:,-1]], axis=3)
        #write_tfboard(other_mask, se, 'Mask Image', 'image')
        write_tfboard(res['target'], se, 'Z Real Image', 'image')
        write_tfboard(res['cmask'], se, 'Target Mask', 'image')
        write_tfboard(res['gen_output'][0], se, 'Target Image', 'image')
        write_tfboard(res['cmask2'], se, 'Gen Mask', 'image')
        write_tfboard(res['gen_output'][1], se, 'Gen Image', 'image')
        write_tfboard(res['disc_real_output'][0], se, 'Disc R Image', 'image')
        write_tfboard(res['disc_generated_output'][0], se, 'Disc G Image', 'image')
        write_tfboard(res['disc_gen2_output'][0], se, 'Disc G2 Image', 'image')
    print()
    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 2 == 0:
      saver.save(S, save_path + str(epoch))
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

# Input pipeline
def get_dataset(path, is_train):
  f = glob.glob(path + '*.npz')
  f = [_ for _ in f if int(_.split('/')[-1].split('.')[0]) < TRAIN_SPLIT]
  d = tf.data.Dataset.from_tensor_slices(f)
  if is_train:
    d = d.repeat(-1)
    d = d.shuffle(BUFFER_SIZE)
  d = d.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  d = d.batch(BATCH_SIZE).make_one_shot_iterator()
  return d, len(f)

def figureify(img_tensor):
  #print(img_tensor.shape)
  plt.figure()
  plt.imshow(img_tensor[0] * 0.5 + 0.5)
  plt.axis('off')
  plt.draw()
  return plt.gcf()

def generate_images(S, model, test_input, tar):
  prediction = model(test_input, training=False)
  plt.figure(figsize=(15,15))

  display_list = S.run([test_input[0], tar[0], prediction[0]])
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def write_tfboard(vals, itr, name, t):
  if t == 'scalar':
    writer.add_scalar('{0}'.format(name), vals, itr)
  elif t == 'histogram':
    writer.add_histogram('{0}'.format(name), vals, itr)
  elif t == 'image':
    #writer.add_figure('{0}'.format(name), figureify(vals), itr, close=False)
    #plt.close()
    writer.add_images('{0}'.format(name), vals, itr, dataformats='NHWC')

# MAIN
S = tf.compat.v1.Session()
with S.as_default():
  # Initialize
  generator = Generator()
  discriminator = Discriminator()
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  checkpoint_dir = './training_checkpoints2'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)
  
  # Train
  train_dataset, STEPS = get_dataset(ZIMGS, True)
  input_image, target, colors, faceid, cmask, mask2, cmask2 = train_dataset.get_next()
  #print("Input shape: {}, Output shape: {}".format(input_image.shape, target.shape))
  op = train_step(input_image, target, colors, faceid, cmask, mask2, cmask2)
  # restoring the latest checkpoint in checkpoint_dir
  ckpt_dir = 'saver_dir_pix_v2_cel/'
  writer = SummaryWriter('{0}/logs'.format(ckpt_dir), flush_secs=30)
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  #if checkpoint:
  #  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  last_epoch = 0
  if ckpt and ckpt.model_checkpoint_path:
    saver = tf.compat.v1.train.Saver(max_to_keep=30,allow_empty=False)
    saver.restore(S, ckpt.model_checkpoint_path)
    last_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('-')[-1])
    print("Restoring from last checkpoint {}".format(last_epoch))
  else:
    S.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(max_to_keep=30,allow_empty=False)
    print("Starting from scratch")
  
  fit(op, EPOCHS, ckpt_dir + '/ckpt-', last_epoch)
  
  
#  TRAIN_OR_TEST = "test"
  #test_dataset, STEPS = get_testset(TE_IMG, False)
# for i in range(5):
#    inp, tar = test_dataset.get_next()
#    generate_images(S, generator, inp, tar)
  # Run the trained model on a few examples from the test dataset
 
