## Introduction

<div align="center">
  <img src="figures/UnitFace_Figure1.pdf" width="600"/>
</div>


| Loss                          |   UT   |   $m$  | $\lambda$ |   $r$  |     MR-All     |      IJB-C     |       LFW      |
|-------------------------------|:------:|:------:|:---------:|:------:|:--------------:|:--------------:|:--------------:|
| $L_{\text{sl}}$               | \xmark | \xmark |   \xmark  | \xmark |      18.52     |      71.53     |      98.30     |
| $L_{\text{bce}}$              | \xmark | \xmark |   \xmark  | \xmark |      18.90     |      69.68     |      98.68     |
| $L_{\text{uce}}$              | \cmark | \xmark |   \xmark  | \xmark |      19.59     |      74.80     |      98.45     |
| $L_{\text{sl-m}}$             | \xmark | \cmark |   \xmark  | \xmark |      41.80     |      46.17     |      99.50     |
| $L_{\text{bce-m}}$            | \xmark | \cmark |   \xmark  | \xmark |      45.35     |      83.88     |      99.46     |
| $L_{\text{uce-m}}$            | \cmark | \cmark |   \xmark  | \xmark |      47.45     |      88.65     | \textbf{99.56} |
| $L_{\text{uce-mb-}{\lambda}}$ | \cmark | \cmark |   \cmark  | \xmark |      48.54     | \textbf{88.96} |      99.55     |
| $L_{\text{uce-mb-}{r}}$       | \cmark | \cmark |   \xmark  | \cmark | \textbf{48.72} |      88.94     |      99.30     |


## Get started

1. **Download dataset**

download [CASIA-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing) preprocessed by [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).


2. **Train model**

unzip faces_webface_112x112.zip

modify the 'data_path' in train.py

choose and replace 'loss' in backbone.py

python train.py

3. **Test model**

python pytorch2onnx.py

zip model.zip model.onnx

upload model.zip to MFR Ongoing
