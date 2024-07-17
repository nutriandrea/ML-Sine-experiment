![image](https://github.com/user-attachments/assets/b68a5270-48b2-421a-8c15-196171945396)![image](https://github.com/user-attachments/assets/1f87b5a5-06ef-4c23-8010-0c9b05f2843d)# ML-Sine-experiment
In this experiment, we have 1000 sample from a sinewave with slight noise. We will train a Neurale Network to predict the signal.

We have 1000 samples from a sinewave with slight noise; 
we will train an NN to predict the signal.
First, we must set up the NN model's development environment.
We need to install the Python toolchain on our computers and install the Tensorflow package for your Python Interpreter. 
Use the Installer program to install Python on your computer.
Type the command 
“python –m pip install tensorflow –i https://pypi.tuna.tsinghua.edu.cn/simple”
in the CMD Console(For better network connection)!

##Data Preparation
-Generate 1000 samples of a sinewave
-Add a slight noise to your samples
-Split the 1000 samples into three parts: Train Set(60%), Validate Set(20%), and Test Set(20%)


##Define A NN and Launch Training
-Add Layers to your model
-Set some Hyperparameters for your model（Batch Size、Epoch）
-Launch Training

##Evaluate your model and Convert it into a TFLite Model
-Make a Prediction with your model and compare it with the Test Dataset
-Quantize your model, Convert it, and save it into a .tflite file
-Finally, convert your model to a C/C++ file


