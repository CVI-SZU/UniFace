## Introduction

## Get started

1. **Download dataset**
download [CASIA-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing) preprocessed by [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).

2. **Train model**
unzip faces_webface_112x112.zip
modify the 'data_path' in train.py
python train.py

3. **Test model**
python pytorch2onnx.py
zip model.zip model.onnx
upload model.zip to MFR Ongoing
