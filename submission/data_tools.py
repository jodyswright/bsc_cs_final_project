# data_tools.py
import numpy as np
import math
# use for splitting the test and train data
from sklearn.model_selection import train_test_split

class Data_Handling:
    '''
    requirements:
        math
        numpy
        train_test_split from sklearn.model_selection
        
    '''
    def __init__(self, output_size=5):
        self.__shape = 0
        self.__output_size = output_size
        self.__length = 0
        
        
    def load_data(self, label_path, data_path, split=0.2):
        # load the images and labels
        self.__labels = np.load(label_path)
        self.__data = np.load(data_path)
        self.__length = self.__labels.shape[0]
        self.__shape = self.__data[1].shape
        

        print("Loaded files of size:")
        print(f"Images: {self.__data.shape}\nLabels: {self.__labels.shape}")
        
    def split_data(self, split=0.2):
        # split and shuffle the data and labels
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__data, self.__labels, test_size=split)

    @property
    def shape(self, ):
        return self.__shape
    
    @property
    def output_size(self, ):
        return self.__output_size
    
    @property
    def X_train(self, ):
        return self.__X_train
    
    @property
    def y_train(self, ):
        return self.__y_train
    
    @property
    def X_test(self, ):
        return self.__X_test
    
    @property
    def y_test(self, ):
        return self.__y_test
    
    @property
    def length(self, ):
        return self.__length