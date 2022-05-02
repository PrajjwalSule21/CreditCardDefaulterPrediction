from Logger.logging import Log_class
from FeatureEngineering.data_scaling_transform import Data_scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelBuilder:

    def __init__(self):
        self.folder = 'F:/Ineuron_Internship/Project_CCDP/Log_Files/'
        self.filename = 'model_building.txt'
        self.df_obj = Data_scaling()
        # self.model_list = []
        # self.dict_model = []
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier()
        self.lgstic = LogisticRegression()

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.log_obj = Log_class(self.folder,self.filename)



    def logictic_regression(self):
        """
                       Method: logictic_regression
                       Description:This method is used to build a base line model called logictic_regression
                       Parameters:None
                       Return: Individual model


                """
        try:
            self.log_obj.log("INFO","Entered into Base line model Logistic Regression")
            X_train,X_test,y_train,y_test = self.df_obj.scaling()

            self.log_obj.log("INFO","Make the model of Logistic Regression")
            model = self.lgstic.fit(X_train, y_train)
            path = r'F:/Ineuron_Internship/Project_CCDP/BaseModel/Basemodel_lg_model.sav'

            self.log_obj.log("INFO","Save the Logistic Regression on Base ModelBuilding directory")
            pickle.dump(model, open(path, 'wb'))
            y_pred = model.predict(X_test)

            self.log_obj.log("INFO","Find out the AccuracyScore, ConfussionMatric and ClasssificationReport of Logistic Regression")
            accuracy = accuracy_score(y_test, y_pred)
            confussion = confusion_matrix(y_test, y_pred)
            classification = classification_report(y_test, y_pred)

            self.log_obj.log("INFO","Save AccuracyScore, ConfussionMatrix and CalssificationReport in txt file")

            file = open(r'F:/Ineuron_Internship/Project_CCDP/BaseModel/score.txt', 'w')
            file.write("Accuracy Score:" + " " + str(accuracy) + '\n')
            file.write("Confusion Matrix:" + '\n' + str(confussion) + '\n')
            file.write("Classification Report:" + " " + str(classification) + '\n')
            file.close()

        except Exception as e:
            self.log_obj.log("Errro on LogisticRegression" + str(e))



    def random_forest(self):

        """
               Method: random_forest
               Description:This method is used to make Random_forest_Classifier to the model
               Parameters:None
               Return:Parameters for Individual model
        """


        try:
            self.log_obj.log("INFO","Entered into best_params_class for random_forest classifier")
            x_train,x_test,y_train,y_test = self.df_obj.scaling()
            self.param_grid = {'criterion': ['gini'], 'max_depth': [890], 'max_features': ['auto'],

                               'min_samples_leaf': [2],
                               # 'min_samples_leaf': [2, 4, 6],

                               'min_samples_split': [2],
                               #'min_samples_split': [0, 1, 2, 3, 4],

                               'n_estimators': [200]
                               #'n_estimators': [600, 700, 800, 900, 1000]

                               }
            self.grid = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, cv=5, n_jobs=-1, verbose=2)
            self.grid.fit(x_train,y_train)

            self.log_obj.log("INFO","Grid_Search cv is performed ")
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.log_obj.log("INFO","Random forest modelling fitting has started")

            self.rfc=RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,
                                            max_depth=self.max_depth,max_features=self.max_features,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_samples_split=self.min_samples_split)
            self.rfc.fit(x_train,y_train)

            self.log_obj.log("INFO",f"Best random forest parameters are {self.grid.best_params_}")
            self.log_obj.log("INFO","Random Forest Modelling completed")

            return self.rfc

        except Exception as e:
            self.log_obj.log("ERROR", "Exception occured at Random forest modelling, Exception is :"+'\n'+str(e))


    def xg_boost(self):

        """
               Method: xg_boost
               Description:This method is used to make xg_boost_Classifier to the model
               Parameters:None
               Return:Parameters for Individual model


        """

        try:
            self.log_obj.log("INFO","Entered into best_params for xg boost class")
            x_train,x_test,y_train,y_test = self.df_obj.scaling()
            self.param_grid_xg = {
                'n_estimators': [100],
                #'n_estimators': [100, 200, 300],

                'learning_rate': [0.01],
                #'learning_rate': [0.5, 0.1, 0.01, 0.001],

                'max_depth': [3]
                #'max_depth': [10, 120, 230, 340]
            }

            self.grid_xg = GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xg,verbose=3, cv=5)
            self.grid_xg.fit(x_train,y_train)

            self.learning_rate = self.grid_xg.best_params_['learning_rate']
            self.n_estimators = self.grid_xg.best_params_['n_estimators']
            self.max_depth = self.grid_xg.best_params_['max_depth']

            self.log_obj.log('INFO',"Xg_boost modelling has started")
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth)

            self.xgb.fit(x_train,y_train)
            self.log_obj.log("INFO",f"Best xg_boost parameters are :{self.grid_xg.best_params_}")
            self.log_obj.log("INFO","XGboost modelling has been completed")

            return self.xgb

        except Exception as e:
            self.log_obj.log("INFO","Occured Exception is:" +'\n'+str(e))


    def get_best_model(self):

        """
                       Method:get_best_model
                       Description:This method is used to get best model from set from the models by comparing their scores.
                       Parameters:None
                       Return: Best model name

        """

        try:
            self.log_obj.log("INFO","Enter get_best_model method in ModelBuilder class ")
            x_train,x_test,y_train,y_test = self.df_obj.scaling()
            self.xgboost = self.xg_boost()
            self.prediction_xgboost = self.xgboost.predict(x_test)

            if len(y_test.unique()) == 1:
                self.xgboost_score = accuracy_score(y_test,self.prediction_xgboost)
                self.log_obj.log("INFO",f"Accuracy of Xgboost model is {self.xgboost_score}")
            else:
                self.xgboost_score = roc_auc_score(y_test,self.prediction_xgboost)
                self.log_obj.log("INFO",f"Roc_Auc score is {self.xgboost_score}")


            self.random_forest=self.random_forest()
            self.prediction_random_forest = self.random_forest.predict(x_test)

            if len(y_test.unique()) == 1:
                self.random_forest_score = accuracy_score(y_test,self.prediction_random_forest)
                self.log_obj.log("INFO",f"Accuracy of random forest is {self.random_forest_score}")
            else:
                self.random_forest_score = roc_auc_score(y_test,self.prediction_random_forest)
                self.log_obj.log("INFO",f"Auc and Roc score of Random forest is {self.random_forest_score}")


            self.log_obj.log("INFO","Selection of best model ")


            if(self.random_forest_score<self.xgboost_score):
                self.best_model_name='xgboost'
                self.best_model=self.xgboost
                self.best_score=self.xgboost_score
            else:
                self.best_model_name='Random_forest'
                self.best_model=self.random_forest
                self.best_score=self.random_forest_score



            self.log_obj.log("INFO","Best model selection is done")
            self.log_obj.log("INFO","Saving model as pickle file")

            path = 'F:/Ineuron_Internship/Project_CCDP/BestModel/'

            if not os.path.isdir(path):
                os.mkdir(path)

            with open(path + self.best_model_name+'.pkl','wb') as file:
                pickle.dump(self.best_model,file)

            self.log_obj.log("INFO",f"Best model saved in file ,Best model is:{self.best_model_name}")

            return self.best_model_name

        except Exception as e:
            self.log_obj.log("Error","Exception occured at get_best_model  is:"+str(e))