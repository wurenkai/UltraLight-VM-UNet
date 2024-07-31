<div align="center">
<h1>UltraLight VM-UNet </h1>
<h3>Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation</h3>

Renkai Wu<sup>1</sup>, Yinghao Liu<sup>2</sup>, Pengchen Liang<sup>1</sup>\*, Qing Chang<sup>1</sup>\*

<sup>1</sup>  Shanghai University, <sup>2</sup>  University of Shanghai for Science and Technology


ArXiv Preprint ([arXiv:2403.20035](https://arxiv.org/abs/2403.20035))


</div>

## 🔥🔥Highlights🔥🔥
### *1.The UltraLight VM-UNet has only 0.049M parameters, 0.060 GFLOPs, and a model weight file of only 229.1 KB.*</br>
### *2.Parallel Vision Mamba (or Mamba) is a winner for lightweight models.*</br>

## News🚀
(2024.04.24) ***The third version of our paper has been uploaded to [arXiv](https://arxiv.org/abs/2403.20035), adding richer experimental validation. These include not limited to:***
- Adding key parameter analysis of Mamba variants.
- Adding experiments on parallel connection of multiple Mamba variants.
- Adding the exploration of plug-and-play PVM Layer.
- Adding more ablation experiments for analysis.

(2024.04.09) ***The second version of our paper has been uploaded to arXiv with adjustments to the description in the methods section.***

(2024.04.04) ***Added preprocessing step for private datasets.***

(2024.04.01) ***The project code has been uploaded.***

(2024.03.29) ***The first edition of our paper has been uploaded to arXiv.*** 📃

### Abstract
Traditionally for improving the segmentation performance of models, most approaches prefer to use adding more complex modules. And this is not suitable for the medical field, especially for mobile medical devices, where computationally loaded models are not suitable for real clinical environments due to computational resource constraints. Recently, state-space models (SSMs), represented by Mamba, have become a strong competitor to traditional CNNs and Transformers. In this paper, we deeply explore the key elements of parameter influence in Mamba and propose an UltraLight Vision Mamba UNet (UltraLight VM-UNet) based on this. Specifically, we propose a method for processing features in parallel Vision Mamba, named PVM Layer, which achieves excellent performance with the lowest computational load while keeping the overall number of processing channels constant. We conducted comparisons and ablation experiments with several state-of-the-art lightweight models on three skin lesion public datasets and demonstrated that the UltraLight VM-UNet exhibits the same strong performance competitiveness with parameters of only 0.049M and GFLOPs of 0.060. In addition, this study deeply explores the key elements of parameter influence in Mamba, which will lay a theoretical foundation for Mamba to possibly become a new mainstream module for lightweighting in the future.

### Different Parallel Vision Mamba （PVM Layer） settings:
| Setting | Briefly | Params | GFLOPs | DSC |
| --- | --- | --- | --- | --- |
| 1 | No paralleling ( Channel number ```C```) | 0.136M | 0.060 | 0.9069 |
| 2 | Double parallel ( Channel number ```(C/2)+(C/2)```) | 0.070M | 0.060 |  0.9073 |
| 3 | Quadruple parallel ( Channel number ```(C/4)+(C/4)+(C/4)+(C/4)```) | 0.049M | 0.060 | 0.9091 |

**0. Main Environments.** </br>
The environment installation procedure can be followed by [VM-UNet](https://github.com/JCruan519/VM-UNet), or by following the steps below (python=3.8):</br>
```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.** </br>
Data preprocessing environment installation (python=3.7):
```
conda create -n tool python=3.7
conda activate tool
pip install h5py
conda install scipy==1.2.1  # scipy1.2.1 only supports python 3.7 and below.
pip install pillow
```

*A. ISIC2017* </br>
1. Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2. Run `Prepare_ISIC2017.py` for data preparation and dividing data to train, validation and test sets. </br>

*B. ISIC2018* </br>
1. Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic18/`. </br>
2. Run `Prepare_ISIC2018.py` for data preparation and dividing data to train, validation and test sets. </br>

*C. PH<sup>2</sup>* </br>
1. Download the PH<sup>2</sup> dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract both training dataset and ground truth folders inside the `/data/PH2/`. </br>
2. Run `Prepare_PH2.py` to preprocess the data and form test sets for external validation. </br>

*D. Prepare your own dataset* </br>
1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './your_dataset/'
  - images
    - 0000.png
    - 0001.png
  - masks
    - 0000.png
    - 0001.png
  - Prepare_your_dataset.py
2. In the 'Prepare_your_dataset.py' file, change the number of training sets, validation sets and test sets you want.</br>
3. Run 'Prepare_your_dataset.py'. </br>

**2. Train the UltraLight VM-UNet.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>

**3. Test the UltraLight VM-UNet.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>

**4. Additional information.** 
- PVM Layer can be very simply embedded into any model to reduce the overall parameters of the model. Please refer to [issue 7](https://github.com/wurenkai/UltraLight-VM-UNet/issues/7) for the methodology of calculating model parameters and GFLOPs. In addition to the above operations, the exact GFLOPs calculation still requires the addition of the SSM values due to the specific nature of SSM. Refer to [here](https://github.com/state-spaces/mamba/issues/110#issuecomment-1919470069) for details. However, due to the small number of UltraLight VM-UNet channels, the addition of all the SSM values has almost no effect on the results of the GFLOPs obtained through the operations described above (3 valid digits).
## Citation
If you find this repository helpful, please consider citing: </br>
```
@article{wu2024ultralight,
  title={UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation},
  author={Wu, Renkai and Liu, Yinghao and Liang, Pengchen and Chang, Qing},
  journal={arXiv preprint arXiv:2403.20035},
  year={2024}
}
```

## Acknowledgement
Thanks to [Vim](https://github.com/hustvl/Vim), [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet) and [LightM-UNet](https://github.com/MrBlankness/LightM-UNet) for their outstanding work.
