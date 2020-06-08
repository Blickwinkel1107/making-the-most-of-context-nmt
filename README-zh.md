# NJUNMT-pytorch-DocNMT

---
[English](README.md), [中文](README-zh.md)
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.com/whr94621/NJUNMT-pytorch.svg?branch=dev-travis-ci)](https://travis-ci.com/whr94621/NJUNMT-pytorch)

NJUNMT-pytorch-DocNMT是论文[“Toward Making the Most of Context in Neural Machine Translation”](https://arxiv.org/abs/2002.07982)的开源实现，是在[NJUNMT-pytorch](https://github.com/whr94621/NJUNMT-pytorch)神经机器翻译工具包的基础上开发而来。

## 目录
- [NJUNMT-pytorch-DocNMT](#njunmt-pytorch-docnmt)
    - [目录](#%E7%9B%AE%E5%BD%95)
    - [依赖的包](#%E4%BE%9D%E8%B5%96%E7%9A%84%E5%8C%85)
    - [使用说明](#%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E)
        - [快速开始](#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
        - [1. 数据预处理](#1-数据预处理)
        - [2. 修改配置文件](#2-%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)
        - [3. 训练](#3-%E8%AE%AD%E7%BB%83)
        - [4. 解码](#4-%E8%A7%A3%E7%A0%81)
    - [和我们联系](#%E5%92%8C%E6%88%91%E4%BB%AC%E8%81%94%E7%B3%BB)

## 依赖包

- python 3.5+
- pytorch 0.4.0+
- tqdm
- tensorboardX
- sacrebleu

## 使用说明

### 快速开始
我们提供了一键在数据集上训练和解码我们模型的脚本。只需要在项目的根目录下执行

``` bash
bash ./scripts/train.sh
```

来进行模型训练，以及执行

``` bash
bash ./scripts/translate.sh
```

在数据集上进行解码。
下面我们将详细说明如何配置训练和解码。

### 1. 数据预处理
#### 1.1 数据获取

本文中所用到的训练集数据如下：
##### ZH-EN
[IWSLT2015 (TED15)](https://wit3.fbk.eu/mt.php?release=2015-01)
##### EN-DE
[IWSLT2017 (TED17)](https://github.com/sameenmaruf/selective-attn/tree/master/data/IWSLT2017)

[News Commentary v11 (News)](http://www.casmacat.eu/corpus/news-commentary.html)

[Europarl v7](https://www.statmt.org/europarl/)
##### EN-RU
[Training data](https://www.dropbox.com/s/5drjpx07541eqst/acl19_good_translation_wrong_in_context.zip)

[Contrastive test sets](https://github.com/lena-voita/good-translation-wrong-in-context/tree/master/consistency_testsets)

Voita等人的详细的篇章现象评估方式和数据处理方式请参考[这里](https://github.com/lena-voita/good-translation-wrong-in-context)

#### 1.2 分词

我们建议使用jieba进行中文分词，使用mosesdecoder自带脚本进行非中文分词。

#### 1.3 字节对编码（可选）

请参考[subword-nmt](https://github.com/rsennrich/subword-nmt)。

#### 1.4 建立词表

为了给源端和目标端的数据建立词表文件，我们提供了一个脚本```./scripts/build_dictionary.py```来建立json格式的词表。

请通过运行:

``` bash
python ./scripts/build_dictionary.py --help
```

来查看该脚本的帮助文件。

我们强烈推荐不要在这里限制词表的大小，而是通过模型的配置文件在训练时来设定。

#### 1.5 文档处理格式

我们模型需要对数据进行文档分界处理，因此需要将数据处理为指定格式。对于一个包含M个文档，并且每个文档含N个句子的文件而言，其格式为：
```
sent1_of_doc1 <EOS> <BOS> sent2_of_doc1 <EOS> <BOS> ... <EOS> <BOS> sentN_of_doc1
sent1_of_doc2 <EOS> <BOS> sent2_of_doc2 <EOS> <BOS> ... <EOS> <BOS> sentN_of_doc2
...
sent1_of_docM <EOS> <BOS> sent2_of_docM <EOS> <BOS> ... <EOS> <BOS> sentN_of_docM
```
考虑到内存有限，我们将文档分割为最多20句子一组，实际上模型支持处理任意句子的文档。
参考格式见[data_format/dev.en.20.sample](data_format/dev.en.20.sample)



### 2. 修改配置文件

可以参考```./configs```文件夹中的一些样例。我们提供了几种配置样例:

- ```ted15_ours.yaml```: 在TED15中英上训练
- ```ted17_ours.yaml```: 在TED17英德上训练
- ```news_ours.yaml```: 在News英德上训练
- ```euro_ours.yaml```: 在Europarl英德上训练


了解更多关于如何配置一个神经机器翻译模型的训练任务，请参考
[这里](https://github.com/whr94621/NJUNMT-pytorch/wiki/Configuration)。

### 3. 训练

通过运行如下脚本来启动一个训练任务

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

执行```python -m src.bin.train --help```来查看更多的选项

训练期间，所有检查点和最好的模型都会被保存在```---saveto```指定的文件夹下面。假设模型的名称被设定为"MyModel"，那么这个文件夹下面会出现如下一些文件：

- **MyModel.ckpt**: 存放了所有保存的检查点名称的文本文件
- **MyModel.ckpt.xxxx**: 在第xxxx步保存的检查点文件
- **MyModel.best**: 存放了所有最好检查点名称的文本文件
- **MyModel.best.xxxx**: 在第xxxx步保存的检查点文件
- **MyModel.best.final**: 最终得到的最好模型文件, 即在验证集上取得最好效果的模型。其中只保留了模型参数

### 4. 解码

当训练结束时，最好的模型会被自动的保存。通常我们只需要用被命名为"xxxx.best.final"的最好模型文件来进行解码。如之前所说，这个模型能在验证集上取得最好的效果

我们可以通过执行下列脚本来进行解码:

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

通过运行```python -m src.bin.translate --help```来查看更多的选项。

同样我们的代码支持集成解码。通过运行```python -m src.bin.ensemble_translate --help```来查看更多的选项。


## 和我们联系

如果你有任何问题，请联系[]()，[yx1107@foxmail.com](mailto:yx1107@foxmail.com)
