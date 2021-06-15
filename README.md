# Insight-face-tf1.13.2
Inference for insight face with tensorflow 1.13.2

![workflow](https://user-images.githubusercontent.com/62841284/121994728-c7008300-cde0-11eb-98b4-3b1a4c835585.jpg)

## Reference
  - 1. https://github.com/deepinsight/insightface
  - 2. https://github.com/bubbliiiing/retinaface-keras 
  - 3. https://github.com/microsoft/MMdnn 
  - 4. https://github.com/Talgin/preparing_data
  
## Index
  - 1. Version Info
  - 2. Training
  - 3. Convert model
  - 4. Inference
  
## Version Info
  - 1. Training insight face
    * Python 3.6.9
    * mxnet 1.5.0
    * Pytorch 1.6.0
    * Torch Vision 0.7.0
    * Cuda 10.1

  - 2. Training retina face and Convert model
    * Python 3.6.9
    * Tensorflow 1.13.2
    * Cuda 10.0
    
## Training
  - 1. preparing data
      * (1) preparing image folders
      * (2) renaming file with 'rename.py'
      * (3) making property
      * (4) making list file 
        * mxnet/tools/im2rec.py
      * (5) making record & index files
        * mxnet/tools/im2rec.py
      * (6) making pairs.txt with 'make_pairs.py'
      * (7) making bin files with 'dataset2bin.py'
![input](https://user-images.githubusercontent.com/62841284/121996231-2a8bb000-cde3-11eb-92f6-c7a04e01c339.jpg)

  - 2. Training 
![train](https://user-images.githubusercontent.com/62841284/121996242-311a2780-cde3-11eb-9ca0-16e0bf870fbd.jpg)

## Convert Model
  - 1. pip install mmdnn
  - 2. mmconvert -sf mxnet -in  path/to/model-symbol.json -iw path/to/model-0000.params -df tensorflow -om path/to/output --inputShape 112,112,3 --dump_tag SERVING

## Inference
![result](https://user-images.githubusercontent.com/62841284/122002077-3e87df80-cdec-11eb-880c-2a33c767a3e3.png)
