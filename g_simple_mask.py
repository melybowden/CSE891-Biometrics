import os
import cv2
import glob
import numpy as np
from utils import make_folder
import pickle
#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
label_short = {
    'skin': ['skin', 'r_ear', 'l_ear', 'neck'],
    'eyes': ['l_eye', 'r_eye'],
    'mouth': ['mouth', 'u_lip', 'l_lip'],
    'hair': ['hair', 'r_brow', 'l_brow'],
    'nose': ['nose'],
    'clothes': ['cloth'],
    'accessories': ['eye_g', 'hat', 'ear_r', 'ear_l', 'neck_l']}
image_folder = 'CelebA-HQ-img'
folder_base = 'CelebAMask-HQ-mask-anno'
folder_save = 'CelebAMask-HQ-mask-anno-short'
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512, len(label_short)))
    # label_colors = np.zeros(len(label_short))
    img_file = os.path.join(image_folder, str(k) + '.jpg')
    # print(img_file)
    if (os.path.exists(img_file)):
        img = cv2.imread(img_file)
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if (os.path.exists(filename)):
                #print (label, idx+1)
                im = cv2.imread(filename)
                im = im[:, :, 0]
                idx = [_ for _,__ in enumerate(label_short) if label in label_short[__]][0]
                im_base[:,:,idx] += im
                #for li, l in enumerate(label_short):
                #    im_base[:,:,li] = im_base[:,:,li] + im if label in label_short[l] else im_base[:,:,li]
                #im_base[im != 0] = (idx + 1)

        img = cv2.resize(img, (256,256))
        im_base = cv2.resize(im_base, (256,256))
        im_base = (im_base >= 1).astype(np.uint8)

        label_colors = np.stack([np.sum(img * im_base[:,:,i:i+1], axis=(0,1)) / (0.0001 + np.sum(im_base[:,:,i])) for i in range(im_base.shape[-1])], 0)
        #print(label_colors.shape)
        #input()
        filename_save = os.path.join(folder_save, str(k) + '.npz')
        print (filename_save)
        np.savez(filename_save, Masks=im_base, Colors=label_colors, Image=img)
#    cv2.imwrite(filename_save, im_base)
