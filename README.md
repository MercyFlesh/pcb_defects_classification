# PCB defects classification

The project implements the task of detection and classification defects in printed circuit boards. A reference based method is used to detect defects, and an ensemble of three CNN models (`VGG16`, `ResNet-101`, `Inception v3`) is made to classify them.

To combine classification models decision scores, a method based on the _fuzzy Choquet integral_ is used.

<div display=flex align-items=stretch>
    <img src="https://user-images.githubusercontent.com/43219252/170744169-2a497b9c-70d7-4097-a8fa-012996f08b72.jpg" alt="temp" width="240" height="240"> 
    <img src="https://user-images.githubusercontent.com/43219252/170746159-1fd35675-9d2b-44dd-b12d-a9d06b29288d.png" alt="temp" width="240" height="240">
</div>
