## Introduction

**This is the official PyTorch implementation of the ICCV 2023 paper.**

[UniFace: Unified Cross-Entropy Loss for Deep Face Recognition.pdf](paper/05138.pdf)

[Supplementary.pdf](paper/05138-supp.pdf)


## Get started

1. **Prepare dataset**

    Download [CASIA-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing) preprocessed by [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).
    ```console
    unzip faces_webface_112x112.zip
    ```

2. **Train model**

    Modify the 'data_path' in train.py (Line 56)

    Select and uncomment the 'loss' in backbone.py (Line 67)
    ```console
    python train.py
    ```

4. **Test model**
    ```console
    python pytorch2onnx.py
    zip model.zip model.onnx
    ```
    Upload model.zip to [MFR Ongoing](http://iccv21-mfr.com/#/leaderboard/academic) and then wait for the results.

    We here provide a pre-trained model ([ResNet-50](paper/model.zip)) for easy and directdevelopment. This model is trained on CASIA-WebFace and achieved 88%+ on IJB-C and 99%+ on LFW.

## Citation

If you find **UniFace** useful in your research, please consider to cite:

  ```bibtex
  @InProceedings{UniFace,
	title = {UniFace: Unified Cross-Entropy Loss for Deep Face Recognition},
	author = {Jiancan Zhou, Xi Jia, Qiufu Li, Linlin Shen, Jinming Duan},
	booktitle = {ICCV},
	year = {2023}
  }
  ```
