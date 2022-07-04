<div align="center">
<img src="images/slogan.png" width=1000 height=200/>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![](https://img.shields.io/github/license/black0017/MedicalZooPytorch)

We strongly believe in open and **reproducible deep learning research**. Our goal is to implement an open-source **tooth volume segmentation library of state of the art 3D deep neural networks in PyTorch**.    
#### Top priorities 22-07
[Update] This [conference paper](https://arxiv.org/abs/2206.08778) has been accepted on 2022 [ICIRA](https://icira2022.org/camera-ready-submission/).
[Update] We will release our dental dataset **CTooth** and more data samples later in these two months. Please follow us and watch this Github repository for releases to be notified. 


### Datasets 

#### Classification from 2D images:
-  [COVID-CT dataset](https://arxiv.org/pdf/2003.13865.pdf)

-  [COVIDx dataset](https://github.com/IliasPap/COVIDNet/blob/master/README.md)

#### 3D dental segmentation dataset
- [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.XqgcL3Uzbmt)


## Latest features (06/2020)

- On the fly 3D total volume visualization
- Tensorboard and PyTorch 1.4+ support to track training progress
- Code cleanup and packages creation
- Offline sub-volume generation 
- Add Hyperdensenet, 3DResnet-VAE, DenseVoxelNet
- Fix mrbrains,Brats2018,Brats2019, Iseg2019, IXI,MICCAI 2019 gleason challenge dataloaders
- Add confusion matrix support for understanding training dynamics
- Some Visualizations


## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated :) !

## Contributing to Medical ZOO
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. More info on the [contribute directory](./contribute/readme.md).

## Current team

#### [Qianni Zhang](https://github.com/QNZhang "Git page")

## License , citation and acknowledgements
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. It would be nice to cite the **original models and datasets**. If you want, you can also **cite this work** as:

```
@MastersThesis{adaloglou2019MRIsegmentation,
author = {Adaloglou Nikolaos},
title={Deep learning in medical image analysis: a comparative analysis of
multi-modal brain-MRI segmentation with 3D deep neural networks},
school = {University of Patras},
note="\url{https://github.com/black0017/MedicalZooPytorch}",
year = {2019},
organization={Nemertes}}
```

####  Acknowledgements
The work was supported by the  National Natural Science Foundation of China under Grant No. U20A20386. Thanks for the data support on the University of Electronic Science and Technology of China and its Hospital.


[contributors-shield]: https://img.shields.io/github/contributors/liangjiubujiu/CTooth.svg?style=flat-square
[contributors-url]: https://github.com/liangjiubujiu/CTooth/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/liangjiubujiu/CTooth.svg?style=flat-square
[forks-url]: https://github.com/liangjiubujiu/CTooth/network/members

[stars-shield]: https://img.shields.io/github/stars/liangjiubujiu/CTooth.svg?style=flat-square
[stars-url]: https://github.com/liangjiubujiu/CTooth/stargazers

[issues-shield]: https://img.shields.io/github/issues/liangjiubujiu/CTooth.svg?style=flat-square
[issues-url]: https://github.com/liangjiubujiu/CTooth/issues
