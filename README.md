# DoubleViT: Pushing transformers towards the end because of convolutions

[![cite-bibtex](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#cite) 

Official repository of the ICPR 2024 paper "DoubleViT: Pushing transformers towards the end because of convolutions"

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-78186-5_11) | [Poster](https://github.com/mahendran-narayanan/Double-ViT/blob/main/docs/icpr24_poster.pdf)

Code contains the DoubleViT model with the CIFAR-10 dataset as example.

[Mahendran Narayanan](https://scholar.google.de/citations?user=c8subicAAAAJ)

## Abstract

Vision transformers have outperformed convolutional networks and dominate the field in vision tasks. Recent trends indicate a shift towards exploring alternatives to attention mechanisms. We introduce DoubleViT, a model that pushes the attention mechanisms towards the end of the network. The network begins with convolutional layers and concludes with attention mechanisms. The convolutional layers and their depth are determined based on input shapes. In this approach, the shift mechanism learns from the outputs of the convolution layers rather than from the input image patches. This fusion enhances the network’s ability to capture better feature representations. This proposed model has a decrease in parameters when compared to other ViTs. We conduct extensive experiments on benchmark datasets to validate the model and compare them with established architectures. Experimental results demonstrate a remarkable increase in the classification accuracy of the proposed model.

## Architecture

![Architecture](https://github.com/mahendran-narayanan/Double-ViT/blob/main/docs/arch_1.svg)


## Citation
If you have used DoubleViT in your research, please cite our work. 🎓

```
@inproceedings{narayanan2024doublevit,
  title={DoubleViT: Pushing Transformers Towards the End Because of Convolutions},
  author={Narayanan, Mahendran},
  booktitle={International Conference on Pattern Recognition},
  pages={157--171},
  year={2024},
  organization={Springer}
}
```