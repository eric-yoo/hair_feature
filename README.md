# simclr-hair
- Unsupervised Learning based pseudo-reconstruction of 3D hair model

## Network
- Model: ResNet50-based encoder network using a novel loss function [reference](https://github.com/google-research/simclr)
- Input: Orientation Map image generated with HairNet preprocessing modules
- Output: vector encoding of the input image

## Training
- The network learns the visual representations in an unsupervised fashion (contrasive learning)
- original data: 343 hair models from [USC Hair salon dataset](http://www-scf.usc.edu/~liwenhu/SHM/database.html with slight rotations and translations rendered in 10 different views

## Hair Model pseudo-reconstruction
- Given an orientation map of a random hair image, our model outputs its vector encoding
- Our module then selects the most similar orientation map out of the pre-calculated 3430 orientation map images
- Our module then returns the 3D hair model corresponding to that orientation map


## Acknowledgement
- [SimCLR version2](https://arxiv.org/abs/2006.10029)
- [A simple framework for contrastive learning of visual representations](https://arxiv.org/abs/2002.05709)
- [SimCLR - A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2006.10029)
