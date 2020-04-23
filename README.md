# Facial Reconstruction and Manipulation GANs
This project uses image-to-image translation to reconstruct and manipulate
face images from semantic segmentation maps to realistic images
using generative adversarial networks (GANs).

This is implemented using TensorFlow and a Pix2Pix-style framework.

## Reconstruction Results
![Facial Reconstruction](https://github.com/melybowden/CSE891-Biometrics/blob/master/econstruction.png)
The top rows are reconstructed, and the bottom are ground truth images.
## Joint Reconstruction and Manipulation Results
![Facial Reconstruction with Manipulation](https://github.com/melybowden/CSE891-Biometrics/blob/master/reconstruction_and_manipulation.png)
Ascending bottom to top: ground truth images, ground truth segmentation maps with average colors from the ground truth image, reconstructed images, new segmentation map with average colors from the ground truth image, manipulated image.

### Sources
    @inproceedings{CelebAMask-HQ,
      title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
      author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2020}
    }

    @article{pix2pix2017,
      title={Image-to-Image Translation with Conditional Adversarial Networks},
      author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
      journal={CVPR},
      year={2017}
    }
