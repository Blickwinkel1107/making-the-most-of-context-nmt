# NJUNMT-pytorch-DocNMT

---
[English](README.md), [中文](README-zh.md)
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.com/whr94621/NJUNMT-pytorch.svg?branch=dev-travis-ci)](https://travis-ci.com/whr94621/NJUNMT-pytorch)

NJUNMT-pytorch-DocNMT is the implementation of paper ["Toward Making the Most of Context in Neural Machine Translation"](https://arxiv.org/abs/2002.07982), and is based on [NJUNMT-pytorch](https://github.com/whr94621/NJUNMT-pytorch), the NMT tool-kit.

## Table of Contents
- [NJUNMT-pytorch-DocNMT](#njunmt-pytorch-docnmt)
    - [Table of Contents](#table-of-contents)
    - [Requirements](#requirements)
    - [Usage](#usage)
        - [0. Quick Start](#0-quick-start)
        - [1. Data Preprocessing](#1-data-preprocessing)
        - [2. Write Configuration File](#2-write-configuration-file)
        - [3. Training](#3-training)
        - [4. Translation](#4-translation)
    - [Contact](#contact)

## Requirements

- python 3.5+
- pytorch 0.4.0+
- tqdm
- tensorboardX
- sacrebleu

## Usage

### 0. Quick Start

We provide push-button scripts to setup training and inference of
our model Corpus. Just execute under root directory of this repo
``` bash
bash ./scripts/train.sh
```
for training and
``` bash
bash ./scripts/translate.sh
```
for decoding.
Detailed setups are as follows.

### 1. Data Preprocessing
#### 1.1 Download Data

Training dataset in paper are listed as follows：
##### ZH-EN
[IWSLT2015 (TED15)](https://wit3.fbk.eu/mt.php?release=2015-01)
##### EN-DE
[IWSLT2017 (TED17)](https://github.com/sameenmaruf/selective-attn/tree/master/data/IWSLT2017)

[News Commentary v11 (News)](http://www.casmacat.eu/corpus/news-commentary.html)

[Europarl v7](https://www.statmt.org/europarl/)

##### EN-RU
[Training data](https://www.dropbox.com/s/5drjpx07541eqst/acl19_good_translation_wrong_in_context.zip)

[Contrastive test sets](https://github.com/lena-voita/good-translation-wrong-in-context/tree/master/consistency_testsets)

Please refer to [here](https://github.com/lena-voita/good-translation-wrong-in-context) to learn how Voita et al. configure and run models on contrastive dataset.

#### 1.2 Tokenization

We suggest using Jieba to tokenize Chinese corpus and use scripts of mosesdecoder to tokenize non-Chinese corpus.

#### 1.3 Byte-Pair Encoding (Optional)

See [subword-nmt](https://github.com/rsennrich/subword-nmt).

#### 1.4 Building Vocabulary

To generate vocabulary files for both source and 
target language, we provide a script in ```./data/build_dictionary.py``` to build them in json format.

See how to use this script by running:
``` bash
python ./scripts/build_dictionary.py --help
```
We highly recommend not to set the limitation of the number of
words and control it by config files while training.

#### 1.5 Documental Data Format for Model Processing

Ours model need to partition data, so the original data need to be processed in a legal format.
The format of a file containing M documents and N sentences in each document is:
```
sent1_of_doc1 <EOS> <BOS> sent2_of_doc1 <EOS> <BOS> ... <EOS> <BOS> sentN_of_doc1
sent1_of_doc2 <EOS> <BOS> sent2_of_doc2 <EOS> <BOS> ... <EOS> <BOS> sentN_of_doc2
...
sent1_of_docM <EOS> <BOS> sent2_of_docM <EOS> <BOS> ... <EOS> <BOS> sentN_of_docM
```
In terms of the limited memory, we partition the original document as up to 20 sentences as a group. In fact our model supports processing any amount of sentences in a document.
Please see [data_format/dev.en.20.sample](data_format\dev.en.20.sample) to learn the sample of data format.

### 2. Write Configuration File

See examples in ```./configs``` folder.  We provide several examples:

- ```ted15_ours.yaml```: run our model on TED15
- ```ted17_ours.yaml```: run our model on TED17
- ```news_ours.yaml```: run our model on News
- ```euro_ours.yaml```: run our model on Europarl

To further learn how to configure a NMT training task, see [this](https://github.com/whr94621/NJUNMT-pytorch/wiki/Configuration) wiki page.

### 3. Training
We can setup a training task by running

``` bash
export CUDA_VISIBLE_DEVICES=0
python -m src.bin.train \
    --model_name <your-model-name> \
    --reload \
    --config_path <your-config-path> \
    --log_path <your-log-path> \
    --saveto <path-to-save-checkpoints> \
    --valid_path <path-to-save-validation-translation> \
    --use_gpu
```

See detail options by running ```python -m src.bin.train --help```.

During training, checkpoints and best models will be saved under the directory specified by option ```---saveto```. Suppose that the model name is "MyModel", there would be several files under that directory:

- **MyModel.ckpt**: A text file recording names of all the kept checkpoints

- **MyModel.ckpt.xxxx**: Checkpoint stored in step xxxx

- **MyModel.best**: A text file recording names of all the kept best checkpoints
  
- **MyModel.best.xxxx**: Best checkpoint stored in step xxxx.
  
- **MyModel.best.final**: Final best model, i.e., the model achieved best performance on validation set. Only model parameters are kept in it.

### 4. Translation

When training is over, our code will automatically save the best model. Usually you could just use the final best model, which is named as xxxx.best.final, to translate. This model achieves the best performance on the validation set.

We can translation any text by running:

``` bash
export CUDA_VISIBLE_DEVICES=0
python -m src.bin.translate \
    --model_name <your-model-name> \
    --source_path <path-to-source-text> \
    --model_path <path-to-model> \
    --config_path <path-to-configuration> \
    --batch_size <your-batch-size> \
    --beam_size <your-beam-size> \
    --alpha <your-length-penalty> \
    --use_gpu
```

See detail options by running ```python -m src.bin.translate --help```.

Also our code support ensemble decoding. See more options by running ```python -m src.bin.ensemble_translate --help```


## Contact

If you have any question, please contact [](), [yx1107@foxmail.com](mailto:yx1107@foxmail.com)
