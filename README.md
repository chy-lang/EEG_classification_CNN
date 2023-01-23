# EEG_classification_CNN
This is the first CNN network I worked on. My teammate developed the structure. I later applied the knowledge I learned in this project into the task CNN_Classification, which I have uploaded onto github.

The model of inception and resnet was adopted, which significantly improved the performance of the net.

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
