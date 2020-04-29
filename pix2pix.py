import tensorflow as tf
from tensorboardX import SummaryWriter
import random
import os
import time
import tensorboard
import datetime
import glob
import numpy as np
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data is preprocessed using CelebAMask Data_preprocessing
# Code based on the official tensorflow Pix2Pix implementation, Image-to-Image
# Translation paper, and MaskGAN paper

PATH = "./"
TR_IMG = "train_img/"
TR_LABEL = "train_label_color/"
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
  """
  Numpy helper function to load anything from files.
  Takes a path to an image and returns the 7D mask, average colors and mask, real
  image, DLIB face ID, and random manipulation masks and calculates the manipulation color
  mask.
  """
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
  """
  TF function that calls NP helper function to load all the images, colors, masks, and face ids.
  """
  masks, colors, image, faceid, cmask, mask2, cmask2 = tf.numpy_function(load_mask, [image_file], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

  return tf.ensure_shape(masks, [256, 256, 7]), tf.reverse(tf.ensure_shape(image, [256, 256, 3]), [-1]), tf.reverse(tf.ensure_shape(colors, [1, 1, 7*3]), [-1]), tf.ensure_shape(faceid, [1,1,128]), tf.ensure_shape(cmask, [256, 256, 3]), tf.ensure_shape(mask2, [256, 256, 7]), tf.ensure_shape(cmask2, [256, 256, 3])

def resize(input_image, real_image, height, width):
  """
  Resizes and input map and real image to be desired size
  """
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  """
  Randomly crops an input map and real image
  """
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  """
  Normalizes colors of an input map and real image between [0,1]
  """
  input_image = input_image / 255
  real_image = real_image / 255

  return input_image, real_image

def random_jitter(input_image, real_image):
  """
  Randomly crops and flips input maps and images as data augmentation
  """
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
  """
  Sets up training data pipeline
  """
  input_image, real_image, colors, faceid, cmask, mask2, cmask2 = load(image_file)
  #input_image, real_image = random_jitter(input_image, real_image)
  colors, real_image = normalize(colors, real_image)

  return input_image, real_image, colors, faceid, cmask, mask2, cmask2

def load_image_test(image_file):
    """
    Sets up testing data pipeline
    """
  input_image, real_image, colors, faceid, cmask, mask2, cmask2 = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  colors, real_image = normalize(colors, real_image)

  return input_image, real_image, colors, faceid, cmask, mask2, cmask2


def downsample(filters, size, apply_batchnorm=True):
  """
  Helper function that defines downsampling layers
  """
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
  """
  Helper function that defines upsampling layers
  """
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
  """
  U-net generator with skip connections. Takes a 7D binary map, color vector,
  average color map, 128D face id vector, manipulation binary map, and manipulation
  average color map as input and generates a realistic reconstruciton and
  reconstruciton in the manipulated pose.
  """
  inputs = tf.keras.layers.Input(shape=[256,256,7])
  # Skin, Eyes, Mouth, Hair, Nose, Clothes, Accessories
  color_inputs = tf.keras.layers.Input(shape=[1, 1, 7*3])
  tar = tf.keras.layers.Input(shape=[256,256,3])
  faceid = tf.keras.layers.Input(shape=[1,1,128])
  mask2 = tf.keras.layers.Input(shape=[256,256,7])
  tar2 = tf.keras.layers.Input(shape=[256,256,3])

  down_stack = [
    downsample(32, 4, apply_batchnorm=False),
    downsample(64, 4),
    downsample(128, 4),
    downsample(256, 4),
    downsample(256, 4),
    downsample(256, 4),
    downsample(256, 4),
    downsample(256, 4),
  ]

  up_stack = [
    upsample(256, 4, apply_dropout=True),
    upsample(256, 4, apply_dropout=True),
    upsample(256, 4, apply_dropout=True),
    upsample(256, 4),
    upsample(128, 4),
    upsample(64, 4),
    upsample(32, 4),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)

  # Select activation function
  #last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='sigmoid')

  x = inputs
  y = mask2
  # Downsampling through the model
  skips = []
  yskips = []
  # concatenate binary and average color maps
  x = tf.concat([x, tar], axis=-1)
  y = tf.concat([y, tar2], axis=-1)
  #x = tf.concat([x, inputs[:,:,:,0:1] * color_inputs[:,:,:,0:3], inputs[:,:,:,3:4] * color_inputs[:,:,:,9:12]], axis=-1) # color for skin and hair
  #y = tf.concat([y, mask2[:,:,:,0:1] * color_inputs[:,:,:,0:3], mask2[:,:,:,3:4] * color_inputs[:,:,:,9:12]], axis=-1) # color for skin and hair
  for down in down_stack:
    x = down(x)
    skips.append(x)
    y = down(y)
    yskips.append(y)
  x = tf.concat([x, color_inputs, faceid], axis=-1) # add on color and ID vectors
  skips = reversed(skips[:-1])
  y = tf.concat([y, color_inputs, faceid], axis=-1) # add on color and ID vectors
  yskips = reversed(yskips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  for up, yskip in zip(up_stack, yskips):
    y = up(y)
    y = tf.keras.layers.Concatenate()([y, yskip])

  # modify last layer based on sigmoid activation
  x = (last(x) - 0.5) * 2
  y = (last(y) - 0.5) * 2
  #x = last(x)

  return tf.keras.Model(inputs=[inputs, color_inputs, tar, faceid, mask2, tar2], outputs=[x, y])

def generator_loss(disc_generated_output, gen_output, target, disc_gen2_output, input_mask, input_mask2):
  """
  Calculate L1 Loss for the reconstruction and create an L1 loss for the manipulation
  and cross entropy GAN loss
  """
  input_mask_binary = tf.cast(tf.reduce_sum(input_mask, axis=-1, keepdims=True) >= 1, tf.float32)
  input_mask2_binary = tf.cast(tf.reduce_sum(input_mask2, axis=-1, keepdims=True) >= 1, tf.float32)

  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  gan_loss += loss_object(tf.ones_like(disc_gen2_output), disc_gen2_output)

  m_overlap = input_mask * input_mask2 # Overlap in masks

  #depthwise summation of overlap -> binary mask for target for gen2
  m_overlap = tf.math.reduce_sum(m_overlap, axis=-1, keepdims=True)

  #same parts to be same
  gen2_l1 = LAMBDA * tf.reduce_mean(tf.abs(m_overlap * target * input_mask2_binary - m_overlap * gen_output[1]))

  # different parts to be different
  inv_overlap = 1 - tf.cast(m_overlap, tf.float32)
  gen2_loss2 = LAMBDA * tf.reduce_mean(tf.clip_by_value(0.08 - tf.abs(inv_overlap * target - inv_overlap * gen_output[1]), 0.04, 0.08))

  l1_loss = LAMBDA * tf.reduce_mean(tf.abs(target * input_mask_binary - gen_output[0]))

  total_gen_loss = gan_loss + l1_loss + gen2_loss2 + gen2_l1

  return total_gen_loss, gan_loss, l1_loss, gen2_l1, gen2_loss2

def Discriminator():
  """
  PatchGAN discriminator (30x30) takes the input binary map (condition) and an image
  generated or real to determine if each patch, and therefore image, is real or fake.
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 7], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output, disc_gen2_output):
  """
  Calculate cross entropy GAN loss
  """
  # Real should be real
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  # Generated images should be fake
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  gen2_loss = loss_object(tf.zeros_like(disc_gen2_output), disc_gen2_output)

  total_disc_loss = real_loss + generated_loss + gen2_loss

  return total_disc_loss

def train_step(input_image, target, colors, faceid, cmask, mask2, cmask2):
  """
  One training step generates a batch of images, feeds through the discriminator,
  calculates losses, and backpropogation.
  Returns output dictionary with generated images and losses
  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # Generate images
    gen_output = generator([input_image, colors, cmask, faceid, mask2, cmask2], training=True)
    # Discriminator w real
    disc_real_output = discriminator([input_image, target], training=True)
    # Disc with reconstruction
    disc_generated_output = discriminator([input_image, gen_output[0]], training=True)
    # Disc with manipulation
    disc_gen2_output = discriminator([mask2, gen_output[1]], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss, gen2_l1, gen2_loss2 = generator_loss(disc_generated_output, gen_output, target, disc_gen2_output, input_image, mask2)

    disc_loss = discriminator_loss(disc_real_output, disc_generated_output, disc_gen2_output)

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
             'gen2_l1':gen2_l1,
             'gen2_loss2':gen2_loss2,
             'cmask2':cmask2
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
        print('\rEpoch: {0}, Step: {1}, GLoss: {2:.2f}, DLoss {3:.2f}, L1 {4:.2f}, Gen2L1 {5:.2f}'.format(epoch, s, res['gen_total_loss'], res['disc_loss'], res['gen_l1_loss'], res['gen2_l1']+res['gen2_loss2']), end='')
        write_tfboard(res['gen_total_loss'], se, 'Gen Total Loss', 'scalar')
        write_tfboard(res['disc_loss'], se, 'Disc Loss', 'scalar')
        write_tfboard(res['gen_l1_loss'], se, 'Gen L1 Loss', 'scalar')
        write_tfboard(res['gen2_l1']+res['gen2_loss2'], se, 'Gen2 L1 Loss', 'scalar')
      if se > 0 and se % 250 == 0:
        #write_tfboard(tf.matmul(tf.constant(res[4]), tf.constant(res[7])), se, 'Input Image', 'image')
        # res[4] 256x256x7 res[7] 7x3
        #other_mask = np.stack([res['mask2'][:,:,:,0], res['mask2'][:,:,:,3], res['mask2'][:,:,:,-1]], axis=3)
        #write_tfboard(other_mask, se, 'Mask Image', 'image')
        write_tfboard(res['target'], se, 'Z Real Image', 'image') #GT
        write_tfboard(res['cmask'], se, 'Target Mask', 'image') # GT Mask
        write_tfboard(res['gen_output'][0], se, 'Target Image', 'image') # reconstruction
        write_tfboard(res['cmask2'], se, 'Gen Mask', 'image') # Manip mask
        write_tfboard(res['gen_output'][1], se, 'Gen Image', 'image') # Manip reconstruction
    print()
    # Save model every 2 epochs
    if (epoch + 1) % 2 == 0:
      saver.save(S, save_path + str(epoch))
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

# Input pipeline
def get_dataset(path, is_train):
  """
  Setsup and returns input pipeline to get dataset from path with one shot iterator
  """
  f = glob.glob(path + '*.npz')
  if is_train:
    f = [_ for _ in f if int(_.split('/')[-1].split('.')[0]) < TRAIN_SPLIT]
  else:
    f = [_ for _ in f if int(_.split('/')[-1].split('.')[0]) >= TRAIN_SPLIT]
  d = tf.data.Dataset.from_tensor_slices(f)
  if is_train:
    d = d.repeat(-1)
    d = d.shuffle(BUFFER_SIZE)
  d = d.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  d = d.batch(BATCH_SIZE).make_one_shot_iterator()
  return d, len(f)

def figureify(img_tensor):
  """
  Use matplotlib to plot image tensor to screen or format for tensorboard
  """
  #print(img_tensor.shape)
  plt.figure()
  plt.imshow(img_tensor[0] * 0.5 + 0.5)
  plt.axis('off')
  plt.draw()
  return plt.gcf()

def generate_images(S, model, test_input, tar):
  """
  Plot Map, GT, and Generated images for simple generation/reconstruction task
  """
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
  """
  Helper for writing to tensorboard
  """
  if t == 'scalar':
    writer.add_scalar('{0}'.format(name), vals, itr)
  elif t == 'histogram':
    writer.add_histogram('{0}'.format(name), vals, itr)
  elif t == 'image':
    #writer.add_figure('{0}'.format(name), figureify(vals), itr, close=False)
    #plt.close()
    writer.add_images('{0}'.format(name), vals, itr, dataformats='NHWC')


def train():
  """
  Setup tf session and intialize generator, discriminator, optimizers, checkpoints
  to save model to periodically.
  Set ckpt_dir to begin from previous model or create a new directory to start from scratch.
  """
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
    op = train_step(input_image, target, colors, faceid, cmask, mask2, cmask2)
    # restoring the latest checkpoint in checkpoint_dir
    ckpt_dir = 'saver_dir/'
    writer = SummaryWriter('{0}/logs'.format(ckpt_dir), flush_secs=30)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
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


def test():
  """
  Create test dataset and generate reconstructions and manipulations saved to
  img_dir using pretrained model at ckpt_dir.
  """
  S = tf.compat.v1.Session()
  with S.as_default():
    # Initialize
    generator = Generator()
    discriminator = Discriminator()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    td, STEPS = get_dataset(ZIMGS, False)
    input_image, target, colors, faceid, cmask, mask2, cmask2 = td.get_next()

    #op = train_step(input_image, target, colors, faceid, cmask, mask2, cmask2)
    img_dir = 'res/'
    # restoring the latest checkpoint
    ckpt_dir = 'saver_dir/'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    last_epoch = 0
    if ckpt and ckpt.model_checkpoint_path:
      saver = tf.compat.v1.train.Saver(max_to_keep=30,allow_empty=False)
      saver.restore(S, ckpt.model_checkpoint_path)
      last_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('-')[-1])
      print("Restoring from last checkpoint {}".format(last_epoch))
      #inputs, color_inputs, tar, faceid, mask2, tar2
      #reconstruction, manipulation
      for s in range(375):
        gen_output = generator([input_image, colors, cmask, faceid, mask2, cmask2], training=True)
        res = S.run([cmask, target, gen_output[0], gen_output[1], cmask2])
        title = ['cmask','real','rec','man','cmask2']

        for i in range(len(res)):
          for _ in range(BATCH_SIZE):
            plt.imsave(img_dir + str(s)+ '_'+title[i]+'_'+str(_)+'.png',res[i][_])
        input_image, target, colors, faceid, cmask, mask2, cmask2 = td.get_next()
    else:
      print("Could not restore from checkpoint.\n")
      return

################################################################################
# MAIN
test()
