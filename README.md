# TFusion: Transformer based N-to-One Fusion Block
```
https://arxiv.org/abs/2208.12776
```
  
Our implementation is on an NVIDIA RTX 3090 (24G) with PyTorch 1.8.1.


## Datasets
We use the BraTS2020 dataset, an open-source dataset.    
Please download and unzip the 'MICCAI_BraTS2020_TrainingData' into `./dataset`.  
Then, please `cd ./process` and run the following commands to prepare the data:
```
python split.py
```

## Training Examples
```
python train.py --phase train --model_name TF_U_Hemis3D
```
Saved models can be found at `./checkpoint`. 
model_name includes :  'TF_U_Hemis3D', 'U_Hemis3D', 'RMBTS', 'TF_RMBTS', 'LMCR', 'TF_LMCR' .

## Test Examples (Please train the model before test.)
```
python train.py --phase test --model_name TF_U_Hemis3D
```
Brain tumor segmentation results for test data can be found at `./checkpoint`.  

## Evaluation
```
python evaluation.py --model_name TF_U_Hemis3D
```

## Swin-TFusion
Swin-TFusion is the version of TFusion based on Swin Transformer, which can be found at `./net`.
Refer to：
```
https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html
```
```
https://github.com/microsoft/Swin-Transformer.
```



