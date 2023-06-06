# Quickstart
> A quickstart guide for nbdev.

================

## Full Nbdev Tutorial
[![youtube](https://i.ytimg.com/vi/l7zS8Ld4_iA/sddefault.jpg)](https://www.youtube.com/watch?v=l7zS8Ld4_iA)

## Install Nbdev to your github repo (For Linux users)
- Clone this repo and your own repo
```bash
git clone this/repo
git clone your/repo
```

- Copy the files from this repo to your repo. Please note that you should not copy the .git folder but you should copy the .gitignore and .github folder.
```bash
cp -r this/repo/* your/repo
cp -r this/repo/.gitignore your/repo
cp -r this/repo/.github your/repo
```

- Install nbdev

Using conda
```bash
conda install nbdev black -c fastai -c conda-forge
nbdev_install_quarto
```
Using pip
```bash
pip black==23.1.0 install nbdev==2.3.12 # Or the latest version pip install -U git+https://github.com/fastai/nbdev.git
nbdev_install_quarto
```

- Change your settings in settings.ini
```bash
# Change all the TODO to your own settings or you can leave it as it is.
```

:collision: **IMPORTANT** You need to change some settings in your github repo to make it work. Please follow the steps below.

In default, the github repo is limited PAT permission to read only. You need to change it to read and write permission. Please follow the steps below.
```bash
# Settings > Actions > General > Workflow permissions > Read and write permission > Save. 
```
Activate github pages in your repo.
```bash
# Setting > Pages > Branch > Choose gh-pages > Save.
```
![active github pages](../assets/active_github_pages.png)

## Usage
- For all the commands, please refer to the [nbdev documentation](https://nbdev.fast.ai/).
- For this repo, all the exported documents are stored in the **docs** folder. Your notebooks which are used to generate the documents are stored in the **nbs** folder.
- When you make changes to your notebooks, and you want to export the documents, please run the following command.
```bash
# Remove old docs and export new docs
rm -rf docs _proc && nbdev_prepare && nbdev_docs
```
- You can organize your page with **nbs/_quarto.yml**. Please refer to the [quarto documentation](https://quarto.org/docs/) for more details.
```bash
contents:
      - section: "Quick start"
        contents:
          - 03_how_to_use_nbdev.ipynb
          - 04_directives.ipynb
          - 05_create_page.ipynb
          - 06_beautiful_page.ipynb
      - text: "---"
      - section: "Examples"
        contents:
          - 00_card_overview.ipynb
          - 01_card.ipynb
          - 02_desk.ipynb
      - text: "---"
      - 07_test.ipynb
```