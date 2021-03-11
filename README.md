# A Residual Dense U-Net for Image Denoising

This repository is for the RDUNet model proposed in the following paper:

[Javier Gurrola-Ramos](https://scholar.google.com.mx/citations?user=NuhdwkgAAAAJ&hl=es), [Oscar Dalmau](https://scholar.google.com.mx/citations?user=5oUOG4cAAAAJ&hl=es&oi=sra) and [Teresa E. Alarcón](https://scholar.google.com.mx/citations?user=gSUClZYAAAAJ&hl=es&authuser=1), ["A Residual Dense U-Net Neural Network for Image Denoising"](https://ieeexplore.ieee.org/document/9360532), IEEE Access, vol. 9, pp. 31742-31754, 2021, doi: [10.1109/ACCESS.2021.3061062](https://doi.org/10.1109/ACCESS.2021.3061062).

## Citation

If you use this paper work in your research or work, please cite our paper:
```
@article{gurrola2021residual,
  title={A Residual Dense U-Net Neural Network for Image Denoising},
  author={Gurrola-Ramos, Javier and Dalmau, Oscar and Alarcón, Teresa E},
  journal={IEEE Access},
  volume={9},
  pages={31742--31754},
  year={2021},
  publisher={IEEE},
  doi={10.1109/ACCESS.2021.3061062}
}
```

![RDUNet](https://github.com/JavierGurrola/RDUNet/blob/main/Figs/Architecture.jpg)

## Pre-trained models

[Link](https://drive.google.com/drive/folders/1jF8YF-7SoVpc4y39_lFl25OBFVQmZAWJ?usp=sharing) to download the pretrained models.

## Dependencies
- Python 3.6
- PyTorch 1.5.1
- pytorch-msssim 0.2.0
- ptflops 0.6.3
- tqdm 4.48.2
- scikit-image 0.17.2
- yaml 0.2.5
- MATLAB (to create testing datasets)

## Dataset
For training, we used [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) dataset. You need to [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/) the dataset for training the model and put the high-resolution image folders in the './Dataset' folder. You can modify the ```train_files.txt``` and ```val_files.txt``` to load only part of the dataset.

## Training
Default parameters used in the paper are set in the ```config.yaml``` file:

```
patch size: 64
batch size: 16
learning rate: 1.e-4
weight decay: 1.e-5
scheduler gamma: 0.5
scheduler step: 3
epochs: 21
```
Additionally, you can choose the device, the number of workers of the data loader, and enable multiple GPU use.

To train the model use the following command:

```python main_train.py```

## Test

Place the pretrained models in the './Pretrained' folder. Modify the ```config.yaml``` file according to the model you want to use: ```model channels: 3``` for the color model and ```model channels: 1``` for the grayscale model.

Test datasets need to be prepared using the MATLAB codes in './Datasets' folder according to the desired noise level. We test the RDUNet model we use the [Set12](https://github.com/cszn/DnCNN/tree/master/testsets), [CBSD68](https://github.com/cszn/FFDNet/tree/master/testsets/CBSD68), [Kodak24](http://r0k.us/graphics/kodak/), and [Urban100](https://drive.google.com/drive/folders/1miKn9Jn-t7w6fg8uGZoRGre1juRQDZVy?usp=sharing) datasets.

To test the model use the following command:

```python main_test.py```

## Results

Results reported in the paper.

Color:
![Color](https://github.com/JavierGurrola/RDUNet/blob/main/Figs/Color.png)

Grayscale:
![Grayscale](https://github.com/JavierGurrola/RDUNet/blob/main/Figs/Grayscale.png)

## Contact

If you have any question about the code or paper, please contact francisco.gurrola@cimat.mx .
