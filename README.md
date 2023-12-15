## Introduction

**This is the official PyTorch implementation of the ICCV 2023 paper.**

[UniFace: Unified Cross-Entropy Loss for Deep Face Recognition.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_UniFace_Unified_Cross-Entropy_Loss_for_Deep_Face_Recognition_ICCV_2023_paper.pdf)

[Supplementary.pdf](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhou_UniFace_Unified_Cross-Entropy_ICCV_2023_supplemental.pdf)


## Get started

**Requirement: [PyTorch](https://pytorch.org/get-started/previous-versions/) >= 1.8.1**

1. **Prepare dataset**

    Download [CASIA-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing) preprocessed by [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).
    ```console
    unzip faces_webface_112x112.zip
    ```

2. **Train model**

    Modify the 'data_path' in [train.py](train.py) (Line 57)

    Select and uncomment the 'loss' in [backbone.py](backbone.py) (Line 67)
    ```console
    python train.py
    ```

4. **Test model**
    ```console
    python pytorch2onnx.py
    zip model.zip model.onnx
    ```
    Upload model.zip to [MFR Ongoing](http://iccv21-mfr.com/#/leaderboard/academic) and then wait for the results.

    We provide a pre-trained model (ResNet-50) on [Google Drive](https://drive.google.com/file/d/1vXZBy_NSG5-jtvsHkoeFVeaepRRE5Mo5/view?usp=drive_link) for easy and direct development. This model is trained on CASIA-WebFace and achieved 48.42% on MR-All and 99.56% on LFW.

## Citation

If you find **UniFace** useful in your research, please consider to cite:

  ```bibtex
  @InProceedings{Zhou_2023_ICCV,
    author    = {Zhou, Jiancan and Jia, Xi and Li, Qiufu and Shen, Linlin and Duan, Jinming},
    title     = {UniFace: Unified Cross-Entropy Loss for Deep Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20730-20739}
  }
  ```
