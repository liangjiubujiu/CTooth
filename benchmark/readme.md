## This is the benchmark for CTooth dataset.


## Explanation on the performances published in the ICIRA.

Existing medical image segmentation methods may perform poorly on the CTooth dataset for several possible reasons:

1. Dataset Specificity: Different medical image datasets have varying characteristics, such as image resolution, noise levels, diversity in lesion types, and shapes. The coronal part of tooth CT images can be segmented effectively with lightweight UNet models after fine-tuning, but identifying tooth roots or small regions with root canal interference remains challenging for state-of-the-art (SOTA) methods. This is one of the motivations for using attention-based approaches.

2. Small Number of Samples: The CTooth dataset has a limited number of annotated samples. In such cases, existing methods may face overfitting issues as models struggle to learn generalization from a restricted dataset.


## Method analysis

If you want to compare your method with the other classical medical segmentation methods I mentioned in the CTooth paper, you can download them in this dir and put them on your method package. All these methods are extracted from a solid GitHub repository(https://github.com/black0017/MedicalZooPytorch). Thanks for their work.
