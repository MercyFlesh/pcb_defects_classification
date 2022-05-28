# PCB defects classification

The project implements the task of detection and classification defects in printed circuit boards. A reference based method is used to detect defects, and an ensemble of three CNN models (`VGG16`, `ResNet-101`, `Inception v3`) is made to classify them.

To combine classification models decision scores, a method based on the _fuzzy Choquet integral_ is used.

## Dataset
For training and test data, an open dataset [DeepPCB](https://github.com/tangsanli5201/DeepPCB) is used that contains scans of pairs images of printed circuit boards in black and white format. All images are in jpeg format and have a size of 640x640. Each pair consists of a template image of the board and a test one.

<div align=center>
    <img src="https://user-images.githubusercontent.com/43219252/170834797-ea3a7441-dbb3-4417-ae71-1184d09aefa2.jpg" alt="temp" width="240" height="240" style="padding: 10px 5px 10px 10px">
    &nbsp;
    <img src="https://user-images.githubusercontent.com/43219252/170834985-5d8220b3-7cd1-402d-8f74-5bc8d3a45cd1.jpg" alt="temp" width="240" height="240" style="padding: 10px 10px 10px 5px">
</div>
<div align=center>
    <b>Template image</b>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <b>Test image</b>
</div>

## Result
The architecture of the ensemble with the fuzzy Choquet integral showed a classification accuracy of 98.6%.

<img src="https://user-images.githubusercontent.com/43219252/170836768-8246c5f7-92f0-471e-b4dc-d4ea8678e4d0.png" alt="temp" width=400 height=360 style="padding-right: 20px;">

```
              precision    recall  f1-score   support

        open      0.995     0.995     0.995       388
       short      0.952     0.997     0.974       301
    mousebit      0.982     0.980     0.981       393
        spur      0.997     0.960     0.978       325
      copper      1.000     1.000     1.000       294
    pin-hole      0.990     0.987     0.988       300

    accuracy                          0.986      2001
   macro avg      0.986     0.986     0.986      2001
weighted avg      0.986     0.986     0.986      2001
```
\
The result of detection and classification of defects, using the module `/tools/defect_detection.py`:

<div align=center>
    <img src="https://user-images.githubusercontent.com/43219252/170744169-2a497b9c-70d7-4097-a8fa-012996f08b72.jpg" alt="temp" width="240" height="240" style="padding: 10px 5px 10px 10px">
    &nbsp;
    <img src="https://user-images.githubusercontent.com/43219252/170746159-1fd35675-9d2b-44dd-b12d-a9d06b29288d.png" alt="temp" width="240" height="240" style="padding: 10px 10px 10px 5px">
</div>
<div align=center>
    <b>Template</b>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <b>Result</b>
</div>
