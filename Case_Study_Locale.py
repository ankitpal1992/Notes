
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB 
from matplotlib import pyplot
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,auc,roc_curve
import statsmodels.formula.api as sm
import scikitplot as skplt
import sys


def mltraining(filepath,filename):
    
    try:
        
        df=pd.read_csv("".join([filepath,filename]))

 #Countplot between Mobile_site_booking and Car_cancellation, we found  car unavailable through mobile site is more compare to cars cancelled     
        sns.countplot(df['mobile_site_booking'],hue=df['Car_Cancellation'],data=df)
        plt.savefig("".join([filepath,'Mobile_site_booking.png']))
        plt.show()

 #Countplot between online_booking and Car_cancellation, we found  car unavailable through mobile site is more compare to cars cancelled     
         
        sns.countplot(df['online_booking'],hue=df['Car_Cancellation'],data=df)
        plt.savefig("".join([filepath,'online_booking.png']))
        plt.show()
 
 #Countplot between travel_type_id and Car_cancellation, we found  car unavailable through mobile site is more compare to cars cancelled     
         
        sns.countplot(df['travel_type_id'],hue=df['Car_Cancellation'],data=df)
        plt.savefig("".join([filepath,'travel_type_id.png']))
        plt.show()
        
        print('how many null values are there in dataset:',df.isnull().sum())

 # using mean function to remove null values in selected columns        
        for i in df[['package_id','to_city_id','from_city_id','to_lat','to_long']]:
            df[i]=df[i].fillna(df[i].mean())
            
 # using mode function to remove null values in selected columns            
            
        for i in df[['from_lat','from_long','from_area_id','to_area_id']]:
            df_null=df[i].value_counts()
            df_null1=df_null.iloc[0:1].keys()
            df1=pd.DataFrame(df_null1)
            null_value=df1.loc[df1.index[0]][0]
            df[i]=df[i].fillna(null_value)
            
 # Converting Categorical to Numerical       
        for i in df.columns:
             if df[i].dtype == 'object':
                    
                df[i]=df[i].astype('category')
                df[i]=df[i].cat.codes
        
        corr=df.corr()   
        
  # Using P value threshold to remove the columns which are highly corelated     

        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
                                        
        selected_columns = df.columns[columns]
                  
        df_sel_fea = df[selected_columns]
            
        selected_columns = selected_columns[1:].values

        def backwardElimination(x, Y, sl, columns):
            numVars = len(x[0])
            for i in range(0, numVars):
                regressor_OLS = sm.OLS(Y, x).fit()
                maxVar = max(regressor_OLS.pvalues).astype(float)
#              print('maxVar',maxVar)
                if maxVar > sl:
                   for j in range(0, numVars - i):
                       if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                          x = np.delete(x, j, 1)
                          columns = np.delete(columns, j)
                   
            regressor_OLS.summary()
            return x, columns
                    
        SL = 0.05
        data_modeled, selected_columns = backwardElimination(df_sel_fea.iloc[:,0:-1].values, df_sel_fea.iloc[:,-1].values, SL, selected_columns)
        
        df_train=df[selected_columns]
        
        x=df_train.drop('Car_Cancellation',axis=1)
        y=df_train['Car_Cancellation']
            
        x_trn,x_tst,y_trn,y_tst=train_test_split(x,y,random_state=100,test_size=0.4)
  
 #Using Logistic Regression

        Logistic=LogisticRegression(C=1,penalty='l2')
        
        logistic_fit=Logistic.fit(x_trn,y_trn)

        logpred=Logistic.predict(x_tst) 
         
        skplt.metrics.plot_confusion_matrix(y_tst, logpred, normalize=True)
        plt.show()

        Logistic_accuracy=accuracy_score(logpred,y_tst)

        Logistic_confusion=confusion_matrix(logpred,y_tst)

        Logistic_classification=classification_report(logpred,y_tst)
   
  #With the Probe function calculateing the score of each record 
    
        proba_Logistic=Logistic.predict_proba(x_tst)
        
        proba_Logistic=proba_Logistic[:,1]
        
        auc_logistic = roc_auc_score(y_tst, proba_Logistic)
        
        fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_tst, proba_Logistic)
        
        fpr_list=list(fpr_logistic)
        
        tpr_list=list(tpr_logistic)
        
        df_score_logistics=pd.DataFrame()
        
        df_score_logistics['False Positive']=fpr_list
        
        df_score_logistics['True Positive']=tpr_list
        
 # Score saved in CSV file of each record True Positive and False positive 

        df_score_logistics.to_csv("".join([filepath,'Logistic Score.csv']))
    
    
 #Saving Accuracy , Confusion_Matrix, Classification_Report in text file of each model

        file1 = open("".join([filepath,"Logistic_Score.txt"]),"w") 
        accuracy= ['Accuracy SCore of Logistic Regression:'] 
        classification=['Classification Report of Logistic Regression:']
        confusion=['Confusion Matrix of Logistic Regression :']

        file1.write("Logistic Regression Output \n \n") 
        file1.writelines(accuracy)
        file1.writelines(str(Logistic_accuracy))

        file1.writelines("\n")
        file1.writelines("\n")
        file1.writelines(classification)
        file1.writelines("\n")
        file1.writelines(str(Logistic_classification))

        file1.writelines("\n")
        file1.writelines(confusion)
        file1.writelines("\n")
        file1.writelines(str(Logistic_confusion))

        file1.close() #to change file access modes 

  #Using RandomForest

        RandomForest=RandomForestClassifier(n_estimators=5,max_depth=4,max_features=4,criterion='gini',random_state=50)
    
        RandomForest.fit(x_trn,y_trn)
        
        randompred=RandomForest.predict(x_tst)
        
        Random_accuracy=accuracy_score(randompred,y_tst)
        
        Random_confusion=confusion_matrix(randompred,y_tst)
        
        Random_classification=classification_report(randompred,y_tst)
        
   #With the Probe function calculateing the score of each record        
        
        proba_RandomForest=RandomForest.predict_proba(x_tst)
        
        proba_RandomForest=proba_RandomForest[: , 1]
        
        roc_auc_score(y_tst, proba_RandomForest)
        
        fpr_RandomForest, tpr_RandomForest, thresholds_RandomForest = roc_curve(y_tst, proba_RandomForest)
        
        fpr_list=list(fpr_RandomForest)
        tpr_list=list(tpr_RandomForest)
        
        df_score_RandomForest=pd.DataFrame()
        
        df_score_RandomForest['False Positive']=fpr_list
        
        df_score_RandomForest['True Positive']=tpr_list
        
   # Score saved in CSV file of each record True Positive and False positive       
        
        df_score_RandomForest.to_csv("".join([filepath,"RandomForest_Score.csv"]))

   #Saving Accuracy , Confusion_Matrix, Classification_Report in text file of each model

        file1 = open("".join([filepath,"RandomForest_Score.txt"]),"w") 
        accuracy= ['Accuracy SCore of RandomForest:'] 
        classification=['Classification Report of RandomForest:']
        confusion=['Confusion Matrix of RandomForest :']

        file1.write("RandomForest Output \n \n") 
        file1.writelines(accuracy)
        file1.writelines(str(Random_accuracy))

        file1.writelines("\n")
        file1.writelines("\n")
        file1.writelines(classification)
        file1.writelines("\n")
        file1.writelines(str(Random_classification))

        file1.writelines("\n")
        file1.writelines(confusion)
        file1.writelines("\n")
        file1.writelines(str(Random_confusion))

        file1.close() #to change file access modes 

  #Using DecisionTree

        Decisiontree=DecisionTreeClassifier(criterion='gini',max_features=6,max_depth=7)
        
        Decisiontree.fit(x_trn,y_trn)
        
        Decisiontreepred=Decisiontree.predict(x_tst)
        
        DecisionTree_accuracy=accuracy_score(Decisiontreepred,y_tst)
        
        DecisionTree_confusion=confusion_matrix(Decisiontreepred,y_tst)
        
        DecisionTree_classification=classification_report(Decisiontreepred,y_tst)
        
   #With the Probe function calculateing the score of each record 

        proba_Decisiontree=Decisiontree.predict_proba(x_tst)
        
        proba_Decisiontree=proba_Decisiontree[: , 1]
        auc_decisiontree=roc_auc_score(y_tst, proba_Decisiontree)
        
        fpr_decision, tpr_decision, thresholds_decision = roc_curve(y_tst, proba_Decisiontree)
        
        fpr_list=list(fpr_decision)
        tpr_list=list(tpr_decision)
        
        df_score_Decision=pd.DataFrame()
        
        df_score_Decision['False Positive']=fpr_list
        
        df_score_Decision['True Positive']=tpr_list
        
   # Score saved in CSV file of each record True Positive and False positive 

        df_score_Decision.to_csv("".join([filepath,'DecisionScore.csv']))
    
  #Saving Accuracy , Confusion_Matrix, Classification_Report in text file of each model  
        
        file1 = open("".join([filepath,"DecisionTree_Score.txt"]),"w") 
        accuracy= ['Accuracy SCore of DecisionTree :'] 
        classification=['Classification Report of DecisionTree:']
        confusion=['Confusion Matrix of DecisionTree :']
  
        file1.write("DecisionTree Output \n \n") 
        file1.writelines(accuracy)
        file1.writelines(str(DecisionTree_accuracy))

        file1.writelines("\n")
        file1.writelines("\n")
        file1.writelines(classification)
        file1.writelines("\n")
        file1.writelines(str(DecisionTree_classification))

        file1.writelines("\n")
        file1.writelines(confusion)
        file1.writelines("\n")
        file1.writelines(str(DecisionTree_confusion))

        file1.close() #to change file access modes 
  
 #Using GaussianNaiveBayes Theorem
    
        Gaussian_Naive=GaussianNB()
        
        Gaussian_Naive.fit(x_trn,y_trn)
        
        Gaussian_pred=Gaussian_Naive.predict(x_tst)
        
        Gaussian_accuracy=accuracy_score(Gaussian_pred,y_tst)
        
        Gaussian_confusion=confusion_matrix(Gaussian_pred,y_tst)
        
        Gaussian_classification=classification_report(Gaussian_pred,y_tst)
        
  #With the Probe function calculateing the score of each record 

        proba_Gaussian_Naive=Gaussian_Naive.predict_proba(x_tst)
        
        proba_Gaussian_Naive=proba_Gaussian_Naive[: , 1]
        auc_Gaussian_Naive=roc_auc_score(y_tst, proba_Gaussian_Naive)
        fpr_Gaussian_Naive, tpr_Gaussian_Naive, thresholds_Gaussian_Naive = roc_curve(y_tst, proba_Gaussian_Naive)

        fpr_list=list(fpr_Gaussian_Naive)
        tpr_list=list(tpr_Gaussian_Naive)
        
        df_score_Gaussian_Naive=pd.DataFrame()
        
        df_score_Gaussian_Naive['False Positive']=fpr_list
        
        df_score_Gaussian_Naive['True Positive']=tpr_list
        
   # Score saved in CSV file of each record True Positive and False positive 

        df_score_Gaussian_Naive.to_csv("".join([filepath,"Gaussian_Naive_Score.csv"]))
   
   #Saving Accuracy , Confusion_Matrix, Classification_Report in text file of each model
        
        file1 = open("".join([filepath,"Gaussian_Naive_Score.txt"]),"w") 
        accuracy= ['Accuracy SCore of Gaussian_Naive:'] 
        classification=['Classification Report of Gaussian_Naive:']
        confusion=['Confusion Matrix of Gaussian_Naive :']
  
        file1.write("Gaussian_Naive Output \n \n") 
        file1.writelines(accuracy)
        file1.writelines(str(Gaussian_accuracy))

        file1.writelines("\n")
        file1.writelines("\n")
        file1.writelines(classification)
        file1.writelines("\n")
        file1.writelines(str(Gaussian_classification))

        file1.writelines("\n")
        file1.writelines(confusion)
        file1.writelines("\n")
        file1.writelines(str(Gaussian_confusion))

        file1.close() #to change file access modes 

    except Error as e :
    
        print ("Error while connecting to MySQL", e)

        
if __name__ == '__main__':
    
    mltraining(*sys.argv[1:])
        
        
        

