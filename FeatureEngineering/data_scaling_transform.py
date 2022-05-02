from Logger.logging import Log_class
from FeatureEngineering.data_split import split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class Data_scaling:

    def __init__(self):
        self.folder='F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename='scaling_tansform.txt'
        self.df_obj=split()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj=Log_class(self.folder,self.filename)

    def train_test_split(self):

        """
                    Method: train_test_split
                    Description: This method is used splitting data into train and test from Dataset
                    Parameters: None
                    Return: train and test variables of Independent and Dependent variables


        """
        try:
            self.log_obj.log("INFO","Start of splitting the data into Train and Test data")
            X,Y = self.df_obj.splitting()
            x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

            self.log_obj.log("INFO","Training and Test data split completed")

            return x_train,x_test,y_train,y_test

        except Exception as e:
            self.log_obj.log("ERROR","Exception is :"+'\t'+str(e))

    def scaling(self):

        """
                    Method: scaling
                    Description: This method is used to scaling the and store it in pickle file
                    Parameters: None
                    Return: scaling data of Independent and dependent variables

        """

        try:
            self.log_obj.log("INFO","Start scaling the data")
            x_train,x_test,y_train,y_test = self.train_test_split()

            std = StandardScaler()
            self.log_obj.log("INFO", "Scaling the data with help of StandardScaler")
            x_train_std = std.fit_transform(x_train)
            x_test_std = std.transform(x_test)

            self.log_obj.log("INFO","Data Scaling completed")

            with open("F:/Ineuron_Internship/Project_CCDP/standard_scalar.pkl",'wb') as file:
                pickle.dump(std,file)

            self.log_obj.log("INFO","Scaling model saved in the folder")

            return x_train_std,x_test_std,y_train,y_test

        except Exception as e:
            self.log_obj.log("ERROR","Exception occured at scaling_splitting  :"+'\t'+str(e))




