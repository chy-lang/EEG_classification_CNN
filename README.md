# EEG_classification_CNN
Course project. EEG signals, which record our brain activities, are voltage sequences recorded from volunteers. Volunteers will conduct different acts during the recording, and the goal is to classify correspondent EEG signals.

This is the first CNN network I worked on. I later applied the knowledge I learned in this project into the task CNN_Classification, which I have uploaded onto github.

The model of inception and resnet was adopted, which significantly improved the performance of the net.

Structure of the network：

ComplexConvNeuralNetwork(

  (conv1): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (incep1): InceptionA(
  
    (branch_pool): Conv2d(10, 24, kernel_size=(1, 1), stride=(1, 1))
    
    (branch1x1): Conv2d(10, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch5x5_1): Conv2d(10, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch5x5_2): Conv2d(16, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    
    (branch3x3_1): Conv2d(10, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch3x3_2): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    (branch3x3_3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    (pool): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    
  )
  
  (conv2): Conv2d(88, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  (incep2): InceptionA(
  
    (branch_pool): Conv2d(20, 24, kernel_size=(1, 1), stride=(1, 1))
    
    (branch1x1): Conv2d(20, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch5x5_1): Conv2d(20, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch5x5_2): Conv2d(16, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    
    (branch3x3_1): Conv2d(20, 16, kernel_size=(1, 1), stride=(1, 1))
    
    (branch3x3_2): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    (branch3x3_3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    (pool): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    
  )
  
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (linear1): Linear(in_features=1408, out_features=128, bias=True)
  
  (linear2): Linear(in_features=128, out_features=3, bias=True)
  
  (conv3): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
)

Inception module:

![image](https://user-images.githubusercontent.com/79852857/213988986-8500fe8a-5f08-4dd7-8d35-1203a3c27fcd.png)

Resnet module:

![image](https://user-images.githubusercontent.com/79852857/213989090-329d367c-eb8a-4e8f-b3d9-b2f266a74bca.png)
