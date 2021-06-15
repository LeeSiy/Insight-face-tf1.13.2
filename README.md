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
    -- 1. Python 3.6.9
    -- 2. mxnet 1.5.0
    -- 3. Pytorch 1.6.0
    -- 4. Torch Vision 0.7.0
    -- 5. Cuda 10.1

  - 2. Training retina face and Convert model
    -- 1. Python 3.6.9
    -- 2. Tensorflow 1.13.2
    -- 3. Cuda 10.0
    
## Training
  - 1. preparing data
      -- 1. preparing image folders
      ex) data/
	          -/label1
		          -image1
		          -image2
	          -/label2
		          -image1
		          -image2
	          -/label3
		          -image1
		          -image2 
	          …
      -- 2. renaming file
      rename files with 'rename.py'
      file name ex)
      label1_0001.jpg
      label2_0002.jpg
      …
      
