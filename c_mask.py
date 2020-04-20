import numpy as np
import glob
from matplotlib import pyplot as plt
#import tensorflow as tf

ldir = "CelebAMask-HQ-mask-anno-short/"
save_dir = "colored_masks/"

def color_masks():
  for i, f in enumerate(glob.glob(ldir + "*.npz")):
    imgd = np.load(f, allow_pickle=True)
    masks = imgd['Masks']
    colors = np.flip((imgd)['Colors'], -1)
    cm = np.matmul(masks,colors)
    #print(cm.shape)
    input_mask_binary = np.sum(masks, axis=-1, keepdims=True)
    #print(input_mask_binary.shape)
    cm = cm/(input_mask_binary + 1e-10)
    np.savez(save_dir + f.split('/')[-1], cmask=cm)
    if i < 5:
      figureify(cm)
      input()
    if i % 100 == 0:
      print(i)
    
def figureify(img):
  #print(img_tensor.shape)
  plt.figure()
  plt.imshow(img/255)
  plt.axis('off')
  plt.draw()
  plt.show()
  
#imgs = np.zeros([256,256,3])  
#for i in range(16):
#  img = np.load(save_dir + str(i) + '.npz')['cmask']
#  imgd = np.load(ldir + str(i) + '.npz', allow_pickle=True)
#  input_mask_binary = np.sum(imgd['Masks'], axis=-1, keepdims=True)
  #print(imgs.shape)
  #print(img.shape)
#  imgs = np.concatenate([imgs, img/input_mask_binary], axis=1)
  #print(img.shape)
  #figureify(img)
  #input()
  
#figureify(imgs)
color_masks()

