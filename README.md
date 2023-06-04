
<h1 align="center">
  3M-SER
  <br>
</h1>

<h4 align="center">Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/3m-ser" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/3m-ser" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/3m-ser" alt="license"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a> •
</p>

## Key Features
- 3M-SER - a multi-modal speech emotion recognition model that uses multi-head attention fusion of multi-feature embeddings to learn the relationship between speech and emotion.
## How To Use
- Clone this repository
```bash
git clone https://github.com/namphuongtran9196/3m-ser.git 
cd 3m-ser
```
- Install requirements
```bash
conda create -n 3m-ser python=3.7 -y
conda activate 3m-ser
pip install -r requirements.txt
```
- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). Or you can download our preprocessed dataset [here](https://drive.google.com/drive/folders/1-0Z3Q4QZ3Z2Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z?usp=sharing).

- Preprocess data
```bash
python preprocess.py --data_path <path_to_iemocap_dataset> --output_path <path_to_output_folder>
```

- Before starting training, you need to modify the config file in the config folder. You can refer to the config file in the config folder for more details.
```bash
python main.py train
```

## Download
- We provide some pre-trained models which achieve the results as in the paper. You can download them [here](https://drive.google.com/drive/folders/1-0Z3Q4QZ3Z2Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z?usp=sharing).
## License
---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;

