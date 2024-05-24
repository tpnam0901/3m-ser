
<h1 align="center">
  3M-SER
  <br>
</h1>

<h4 align="center">Official code repository for paper "Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings". Paper accepted to EAI INISCOM 2023</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/3m-ser?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/3m-ser?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/3m-ser?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">4M-SER</a> •
  <a href="#abstract">Abstract</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>
## 4M-SER
- You can found our extend 3M-SER [here](https://github.com/namphuongtran9196/4m-ser) - 4M-SER:Comprehensive Study of Multi-Feature Embeddings and Multi-Loss Functions with Multi-Head Self-Attention Fusion for Multi-Modal Speech Emotion Recognition.
## Abstract
> Recent research has shown that multi-modal learning is a successful method for enhancing classification performance by mixing several forms of input, notably in speech-emotion recognition (SER) tasks. However, the difference between the modalities may affect SER performance. To overcome this problem, a novel approach for multi-modal SER called 3M-SER is proposed in this paper. The 3M-SER leverages multi-head attention to fuse information from multiple feature embeddings, including audio and text features. The 3M-SER approach is based on the SERVER approach but includes an additional fusion module that improves the integration of text and audio features, leading to improved classification performance. To further enhance the correlation between the modalities, a LayerNorm is applied to audio features prior to fusion. Our approach achieved an unweighted accuracy (UA) and weighted accuracy (WA) of 79.96% and 80.66%, respectively, on the IEMOCAP benchmark dataset. This indicates that the proposed approach is better than SERVER and recent methods with similar approaches. In addition, it highlights the effectiveness of incorporating an extra fusion module in multi-modal learning.
## Key Features
- 3M-SER - a multi-modal speech emotion recognition model that uses multi-head attention fusion of multi-feature embeddings to learn the relationship between speech and emotion.
## How To Use
- Clone this repository 
```bash
git clone https://github.com/namphuongtran9196/3m-ser.git 
cd 3m-ser
```
- Create a conda environment and install requirements
```bash
conda create -n 3m-ser python=3.8 -y
conda activate 3m-ser
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). Or you can download our preprocessed dataset [here](https://github.com/namphuongtran9196/3m-ser-private/releases).

- Preprocess data
```bash
cd scripts && python preprocess.py --data_root <path_to_iemocap_dataset> --output_dir <path_to_output_folder>
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.
```bash
cd scripts && python train.py -cfg <path_to_config_file>
```

- You can visualize the confusion matrix of the model and other metrics on the test set by following the instructions in the [notebook](./src/visualization/metrics.ipynb).

- You can also find our pre-trained models in the [release](https://github.com/namphuongtran9196/3m-ser/releases).

## Download
- We provide some pre-trained models which achieve the results as in the paper. You can download them [here](https://github.com/namphuongtran9196/3m-ser/releases).
## License
- We use the [Unlicense](https://unlicense.org/) license. You can use it for any purpose.

## Citation
If you use this code or part of it, please cite our work. On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository" button. Or, see the comments in CITATION.cff for the raw BibTeX.

```bibtex
#BIB
@InProceedings{10.1007/978-3-031-47359-3_11,
author="Tran, Phuong-Nam
and Vu, Thuy-Duong Thi
and Dang, Duc Ngoc Minh
and Pham, Nhat Truong
and Tran, Anh-Khoa",
editor="Vo, Nguyen-Son
and Tran, Hoai-An",
title="Multi-modal Speech Emotion Recognition: Improving Accuracy Through Fusion of VGGish and BERT Features with Multi-head Attention",
booktitle="Industrial Networks and Intelligent Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="148--158",
abstract="Recent research has shown that multi-modal learning is a successful method for enhancing classification performance by mixing several forms of input, notably in speech-emotion recognition (SER) tasks. However, the difference between the modalities may affect SER performance. To overcome this problem, a novel approach for multi-modal SER called 3M-SER is proposed in this paper. The 3M-SER leverages multi-head attention to fuse information from multiple feature embeddings, including audio and text features. The 3M-SER approach is based on the SERVER approach but includes an additional fusion module that improves the integration of text and audio features, leading to improved classification performance. To further enhance the correlation between the modalities, a LayerNorm is applied to audio features prior to fusion. Our approach achieved an unweighted accuracy (UA) and weighted accuracy (WA) of 79.96{\%} and 80.66{\%}, respectively, on the IEMOCAP benchmark dataset. This indicates that the proposed approach is better than SERVER and recent methods with similar approaches. In addition, it highlights the effectiveness of incorporating an extra fusion module in multi-modal learning.",
isbn="978-3-031-47359-3"
}

```
## References

[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;
