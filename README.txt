train_test_builder: need to run before project. This is file that generate the training data set. The parameters can be changed within the
	the file, such as the range of data size, the stride, etc.

main: the basic file to run the training and evaluation process. It utilize the package "mitcv" which is a PyTorch implementation of semantic
	segmentation model from MIT CV group. The instruction to run the training and evaluation process can be viewed in README file under 
	that package folder.

config: constains the configuration files of segmentation models. The parameters can be changed within the file for different types of model
