## This is the benchmark for CTooth dataset.


## Explanation for the segmentation performances published in the ICIRA.

Existing medical image segmentation methods may perform poorly on the CTooth dataset for several possible reasons:

Dataset Specificity: Different medical image datasets have varying characteristics, such as image resolution, noise levels, diversity in lesion types, and shapes. The coronal part of tooth CT images can be segmented effectively with lightweight UNet models after fine-tuning, but identifying tooth roots or small regions with root canal interference remains challenging for state-of-the-art (SOTA) methods. This is one of the motivations for using attention-based approaches.

Small Sample Problem: The CTooth dataset has a limited number of annotated samples. In such cases, existing methods may face overfitting issues as models struggle to learn generalization from a restricted dataset.
