# Hands-On Transfer Learning with Python
# 파이썬을 활용한 딥러닝 전이학습


책 안내
https://wikibook.co.kr/tag/%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5/

책 관련 발표 동영상

https://www.youtube.com/playlist?list=PLqkITFr6P-oQlgZS4-XNRcYW2FBUPrc0h

The code will look like the following:
```
import glob
import numpy as np
import os
import shutil
from utils import log_progress
np.random.seed(42)
```


## 소프트웨어와 하드웨어 리스트

Anaconda를 이용하여 라이브러리들을 설치하고, 코드들의 검증 작업을 진행하였습니다. 

* Python 3.6 or 3.7.1, 
* TensorFlow 1.10 or 1.13, 
* Keras 2.2.0 이상
* GraphViz
* Windows, Mac OS X, and Linux (Any)
              
### 필요 Python 패키지

pip install xlrd seaborn matplotlib numpy scipy sklearn jupyter pydot gensim bs4

### 원본 Source Repository와의 차이점

1. 업데이트 된 버전의 tensorflow와 keras를 사용하기로 하였습니다. 그에 따라서 ipynb의 파일들이 책 또는 원본 Repository와 다를 수 있습니다.
2. Keras 사용 부분의 경우 tensorflow의 keras 패키지(tf.keras)를 사용하도록 변경하였습니다. 
3. ipynb에서 사용한 '%matplotlib inline'의 위치를 ipynb에 처음에 나오도록 바꾸었습니다.

## Anaconda에 대해서 

Anaconda는 Data Science 및 Scientific Computing을 위해서 필요한 프로그램, 라이브러리 및 패키지 등을 OS에 관계없이 동일하게 제공하는데 중점을 둔 프로그램입니다. 예를 들어 GPU상에서 동작하는 Tensorflow를 설치하기 위해서는 (책에서 언급하였듯이) 많은 노력이 필요합니다. Graphic Driver및 CUDA Driver등을 설치하는 작업부터 Tensorflow를 GPU에 맞게 빌드하는 작업까지를 수작업으로 하여야 하는 문제등은 이쪽 분야를 하려는 초보자들에게는 어려운 일들입니다. Anaconda에서는 'conda'라는 명령어를 사용하여 책에서 언급한 수많은 내용들을 내부적으로 처리하는 역할을 대신 수행합니다. 

Anaconda를 설치하는 방법은 Command Line방식과 GUI를 이용한 설치방법이 있습니다. Windows나 Mac을 이용하는 경우는 Graphical Installer를 사용하여 설치하는 것이 쉬울 수 있습니다. Linux의 경우는 배포판에 따라서 GUI가 다르기 떄문에 Graphical Installer가 제공되지 않습니다.

### Command Line에서 설치방법

Graphical Installer의 설치 방법은 쉽기 때문에 여기서 넘어가기로 합니다.

1. Anaconda 홈페이지(https://www.anaconda.com/distribution/#download-section)에서 자신의 CPU에 맞는 파일을 다운로드 받습니다. 대부분의 컴퓨터가 x86계열의 CPU를 사용하므로 "64-Bit (x86) Installer"를 다운로드 받습니다. 

2. terminal을 열어서 다음과 같이 입력합니다. root 권한에서 작업할 필요는 없습니다. Anaconda자체가 개인 계정에서도 모두 다 실행될 수 있도록 내부적으로 구성되어 있습니다.

```
$ bash <Download 파일 이름>
```

(현재 최신 Anaconda 버젼의 파일이름은 "Anaconda3-2019.03-Linux-x86_64.sh"입니다.)

3. 라이센스 및 설치 위치등을 물어보는 것들이 나오는데, 기본값으로 설치합니다.

### 설치 확인

Anaconda가 정상적으로 설치되었는지 확인하기 위한 한가지 방법은 Windows의 경우 Command Line창을, Mac과 Linux의 경우는 Terminal창을 열어서 다음과 같이 입력하면 됩니다.


* Windows의 경우:
```
(base) C:\Users\user>where conda
```

* Mac 또는 Linux의 경우:
```
$ which conda
```

정상적으로 설치되었을 경우 conda파일이 설치된 경로가 나타날 것입니다.

사용자 계정의 이름이 user로 되어 있을 경우, 위 명령어 실행시 다음과 같이 결과가 나올 가능성이 높습니다.


* Windows의 경우:
```
C:\Users\user\anaconda3
```

* Mac의 경우:
```
\Users\user\anaconda3
```

* Linux의 경우: 
```
\home\user\anaconda3
```


위 절차가 마무리 되면 다음과 같이 입력하여 Tensorflow및 Graphviz를 설치할 수 있습니다.

```
conda install tensorflow graphviz
```

그 외 나머지 것들은 문서에 적혀있는데로 pip(python3의 경우 pip3)를 이용하여 설치하면 됩니다.

**참고자료**

윈도우에 Anaconda 및 Tensorflow설치하기 : https://tensorflow.blog/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%A6%88%EC%97%90-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0/
Linux(Ubuntu 18.04)에서 Anaconda 및 Tensorflow설치하기 : https://devlikeman.tistory.com/4

**(Optional) IDE 설치**

Windows나 Mac을 사용하고, 기존에 사용하던 편집기가 불편하다고 생각된다면, PyCharm IDE의 사용을 생각해볼 수 있습니다. PyCharm은 Community Edition과 Professional Edition이 존재하는데, 둘의 차이는 Jupyter notebook과 Scientific Computing관련 통합환경을 제공하느냐 아니냐의 차이입니다. 또 한가지는 돈을 내느냐 아니냐가 있습니다!!

Professional Edition에서 몇 개의 기능을 제외한 Community Edition은 상업적인 용도를 제외하고는 자유롭게 다운로드 받아서 Python 개발에 사용할 수 있는 버전입니다.

PyCharm의 설치는 간단합니다. 아래 링크에 들어가서 "PyCharm for Anaconda Community Edition" 버전 다운로드한 후 해당 바이너리 파일을 실행하면 됩니다.

https://www.jetbrains.com/pycharm/promo/anaconda/

## 각 챕터별 데이터 설치시 주의할 점

### Chapter 7

Chapter07 폴더 밑에 data라는 폴더를 만드신 후 aclImdb_v1.tar.gz파일의 압축을 해제하시면 됩니다. 예를 들어, 리눅스에서 Chapter07폴더에 있다면, 다음과 같이 입력할 수 있습니다. (단, Chapter07 폴더에 aclImdb_v1.tar.gz와 amazonreviews.zip이 있다고 가정하겠습니다.)

```
$ mkdir data && cd data
$ tar xvfz ../aclImdb_v1.tar.gz
```

만일 폴더의 위치를 바꾸고 싶다면, config.py파일의 내용을 수정하셔야 합니다. 저자는 데이터 폴더의 구조를 다음과 같이 가정하고 있습니다.

```
TEXT_DATA_DIR -+- IMDB_DATA
               |
               +- IMDB_DATA_CSV
               |
               +- PROCESSED_20_NEWS_GRP
               |
               +- AMAZON_TRAIN_DATA
               |
               +- AMAZON_TEST_DATA
               |
               +- GLOVE_DIR
               |
               +- WORD2VEC_DIR
```

위 구조를 참조하여 적절히 수정하셔야 합니다. 
예를들어, 다음과 같이 데이터 구조를 만들고 싶을 경우


```
data -+- imdb -+- data
      |        |
      |        +- csv
      |
      +- 20newsgrp
      |
      +- amazon_review -+- train
      |                 |
      |                 +- test
      |
      +- glove
      |
      +- word2vec
```

다음과 같이 config.py를 수정할 수 있습니다.

```python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:06:03 2018

@author: tghosh
"""

import pandas as pd
import numpy as np
import os

TEXT_DATA_DIR = './data/'

#Dataset from http://ai.stanford.edu/~amaas/data/sentiment/
IMDB_DATA = TEXT_DATA_DIR + 'imdb/data'
IMDB_DATA_CSV = TEXT_DATA_DIR + 'imdb/csv'

PROCESSED_20_NEWS_GRP = TEXT_DATA_DIR + '20newsgrp'

AMAZON_TRAIN_DATA = TEXT_DATA_DIR+'amazon_review/train.ft'
AMAZON_TEST_DATA = TEXT_DATA_DIR+'amazon_review/test.ft'

GLOVE_DIR = TEXT_DATA_DIR+ 'glove'
WORD2VEC_DIR = TEXT_DATA_DIR+ 'word2vec'

MODEL_DIR = './checkpoint'
```
