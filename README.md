
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
  <a href="#abstract">Abstract</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> Recent research has shown that multi-modal learning is a successful method for enhancing classification performance by mixing several forms of input, notably in speech-emotion recognition (SER) tasks. However, the difference between the modalities may affect SER performance. To overcome this problem, a novel approach for multi-modal SER called 3M-SER is proposed in this paper. The 3M-SER leverages multi-head attention to fuse information from multiple feature embeddings, including audio and text features. The 3M-SER approach is based on the SERVER approach but includes an additional fusion module that improves the integration of text and audio features, leading to improved classification performance. To further enhance the correlation between the modalities, a LayerNorm is applied to audio features prior to fusion. Our approach achieved an unweighted accuracy (UA) and weighted accuracy (WA) of 66.88% and 67.21%, respectively, on the IEMOCAP benchmark dataset. This indicates that the proposed approach is better than SERVER and recent methods with similar approaches. In addition, it highlights the effectiveness of incorporating an extra fusion module in multi-modal learning.
## Key Features
- 3M-SER - a multi-modal speech emotion recognition model that uses multi-head attention fusion of multi-feature embeddings to learn the relationship between speech and emotion.
## How To Use
- Clone this repository 
```bash
git clone https://github.com/namphuongtran9196/3m-ser-private.git 
cd 3m-ser
```
- Create a conda environment and install requirements
```bash
conda create -n 3m-ser python=3.8 -y
conda activate 3m-ser
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```
- There are some error with torchvggish when using with GPU. Please change the code at line 58,59 in file torchvggish/model.py to accept the GPU:
```python
        self._pca_matrix = torch.as_tensor(params["pca_eigen_vectors"]).float().cuda()
        self._pca_means = torch.as_tensor(params["pca_means"].reshape(-1, 1)).float().cuda()
        # Or you can set somethings like .to(device) to make it flexible
```
- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). Or you can download our preprocessed dataset [here](https://github.com/namphuongtran9196/3m-ser-private/releases).

- Preprocess data
```bash
cd scripts && python preprocess.py --data_root <path_to_iemocap_dataset> --output_dir <path_to_output_folder>
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.
```bash
cd scripts && python main.py train
```

## Download
- We provide some pre-trained models which achieve the results as in the paper. You can download them [here](https://github.com/namphuongtran9196/3m-ser-private/releases).
## License
- We use the [Unlicense](https://unlicense.org/) license. You can use it for any purpose.

## Citation
If you use this code or part of it, please cite the following papers:
```
@inproceedings{

}
```

## References

[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;