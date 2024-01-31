#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

#get features from the file features.txt
features=list()
with open("C:\\Users\\91630\\Downloads\\human+activity+recognition+using+smartphones\\UCI HAR Dataset\\UCI HAR Dataset\\features.txt") as f:
    features=[line.split()[1] for line in f.readlines()]
print('no of features:{}'.format(len(features)))


# # GET THE TRAIN DATA

# In[20]:


#get the data from txt files to pandas dataframes
x_train=pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt',delim_whitespace=True,header=None)
x_train.columns=[features]
#add subject column to the dataframes
x_train['subject']=pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\subject_train.txt',header=None).squeeze="columns"
y_train=pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt',names=['Activity']).squeeze("columns")
y_train_labels=y_train.map({1:'WALKING',2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                           4:'SITTING',5:'STANDING',6:'LAYING'})
#Put all columns in single dataframes
train=x_train
train['Activity']=y_train
train['ActivityName']=y_train_labels
train.sample(2)


# In[21]:


train.shape


# In[5]:


x_test = pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\X_test.txt', delim_whitespace=True, header=None)
x_test.columns = [features]
y_test = pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt', names=['Activity']).squeeze("columns")
#add subject column to the dataframes
x_test['subject']=pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\subject_test.txt',header=None).squeeze="columns"
y_test=pd.read_csv(r'C:\Users\91630\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt',names=['Activity']).squeeze("columns")
y_test_labels=y_test.map({1:'WALKING',2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                           4:'SITTING',5:'STANDING',6:'LAYING'})
# Put all columns in a single dataframe for test
test = x_test
test['Activity'] = y_test
test['ActivityName'] =y_test_labels
test.sample(2)


# In[11]:


test.shape


# # DATA CLEANING
# 

# In[6]:


print('no of duplicates in train:{}'.format(sum(train.duplicated())))
print('no of duplicates in test:{}'.format(sum(test.duplicated())))


# In[7]:


print('{} NAN/NULL values in train'.format(train.isnull().values.sum()))
print('{} NAN/NULL values in test'.format(test.isnull().values.sum()))


# In[8]:


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)


# # EDA

# In[9]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.shape,test.shape)


# In[10]:


import numpy as np
import pandas as pd
columns=train.columns
# Assuming 'columns' is a pandas Series or DataFrame column
columns = columns.str.replace('[()]', '', regex=True)
columns = columns.str.replace('[-]', '', regex=True)
columns = columns.str.replace('[,]', '', regex=True)
train.columns=columns
test.columns=columns

test.columns


# # activity duration

# In[27]:


pip install plotly


# In[12]:


import plotly
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
label_counts=train['ActivityName'].value_counts()
n=label_counts.shape[0]
colormap=plt.get_cmap('viridis')
colors = [to_hex(colormap(col)) for col in np.arange(0, 1.01, 1 / (n - 1))]
data=go.Bar(x = label_counts.index,
           y=label_counts,marker=dict(color=colors))
layout=go.Layout(title='Smartphone ActivityName Distribution',
                xaxis=dict(title='ActivityName'),
                yaxis=dict(title='count'))
fig=go.Figure(data=[data],layout=layout)
fig.show()





# # static and dynamic are different

# In[13]:


sns.set_palette("Set1",desat=0.80)
facetgrid=sns.FacetGrid(train,hue='ActivityName',height=5,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMagmean',hist=False)\
    .add_legend()
plt.annotate("Stationary Activites",xy=(-0.960,12),xytext=(-0.5,15),size=20,\
            va='center',ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.annotate("Moving Activites",xy=(0,3),xytext=(0.2,9),size=20,\
            va='center',ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show()


# In[14]:


#magnitude of an acceleration will separate good

plt.figure(figsize=(7,5))
sns.boxplot(x='ActivityName',y='tBodyAccMagmean',data=train,showfliers=False,saturation=1)
plt.ylabel('Acceleration magnitude mean')
plt.axhline(y=-0.7,xmin=0.1,xmax=0.9,dashes=(5,5),c='g')
plt.axhline(y=-0.05,xmin=0.4,dashes=(5,5),c='m')
plt.xticks(rotation=90)
plt.show()


# In[15]:


sns.boxplot(x='ActivityName',y='angleXgravityMean',data=train)
plt.axhline(y=0.08,xmin=0.1,xmax=0.9,c='m',dashes=(5,3))
plt.title('Angle between xaxis and Gravity_mean',fontsize=15)
plt.xticks(rotation=40)
plt.show()
            


# In[16]:


sns.boxplot(x='ActivityName',y='angleYgravityMean',data=train,showfliers=False)
plt.title('Angle between yaxis and Gravity_mean',fontsize=15)
plt.xticks(rotation=40)
plt.axhline(y=-0.22,xmin=0.1,xmax=0.8,dashes=(5,3),c='m')
plt.show()


# In[17]:


import numpy as np

import pandas as pd

import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from datetime import datetime

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier


# In[18]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.shape,test.shape)


# In[19]:


train.head(5)


# In[20]:


X_train= train.drop(['subject', 'Activity', 'ActivityName'], axis=1) 
y_train =train.ActivityName

X_test=test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
y_test=test.ActivityName

print('X_train and y_train: ({},{})'.format(X_train.shape, y_train.shape)) 
print('X_test and y_test: ({},{}'.format(X_test.shape, y_test.shape))


# In[21]:


labels=['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING UPSTAIRS']


# In[47]:


plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm=cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)


    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks (tick_marks, classes)

    fnt='.2f' if normalize else 'd'

    thresh=cm.max()/2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format (cm[i, j], fnt),
                horizontalalignment="center",
                color="white" if cm[i, j]> thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel("Predicted label")


# In[48]:


from datetime import datetime

def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, 
                  print_cm=True, cm_cmap=plt.cm.Greens):
    results = dict()
    train_start_time= datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done....!\n')
    train_end_time = datetime.now()
    results['training_time'] = train_end_time - train_start_time
    print('==> training time: {}\n'.format(results['training_time']))
    print('Predicting test data')
    test_start_time=datetime.now()
    y_pred =model.predict(X_test)
    test_end_time= datetime.now()
    print('Done.....\n')
    results['testing_time'] = test_end_time - test_start_time
    print('==> testing time: {}\n'.format(results['testing_time']))
    results['predicted']=y_pred
    accuracy=metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
    results['accuracy']=accuracy
    print('==> Accuracy:{}\n'.format(accuracy))
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm:
        print('\n****Confusion Matrix*****')
        print('\n {}'.format(cm))
    plt.figure(figsize=(6,6))
    plt.grid(False)
        
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap=cm_cmap)
    plt.show()
    print("******* Classifiction Report *****")
    classification_report = metrics.classification_report(y_test, y_pred)
    results['classification_report']=classification_report
    print(classification_report)
    results['model']=model
    return results


# In[49]:


def print_grid_search_attributes(model):
    print('\n\n==> Best Estimator:')
    print('\t{}\n'.format(model.best_estimator_))
    print('\n==> Best parameters:')
    print('\tParameters of best estimator:{}'.format(model.best_params_))
    print('\n==> No. of CrossValidation sets:')
    print('\tTotal numbre of cross validation sets: {}'.format(model.n_splits_))
    print('\n==> Best Score:')
    print('\tAverage Cross Validate scores of best estimator {}'.format(model.best_score_))


# # logistic regression
# 

# In[50]:


import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
log_reg=linear_model.LogisticRegression()
log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
log_reg_grid_results=perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)
print_grid_search_attributes(log_reg_grid_results['model'])


# # linear svc with gridsearch

# In[51]:


parameters = {'C': [0.125, 0.5, 1, 2, 8, 16]}

lr_svc= LinearSVC(tol=0.00005)

lr_svc_grid=GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)

lr_svc_grid_results=perform_model(lr_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model

print_grid_search_attributes(lr_svc_grid_results['model'])


# # kernal svm with gridSearch

# In[55]:


parameters = {'C': [2,8,16],\
              'gamma': [ 0.0078125, 0.125, 2]}

rbf_svm =SVC (kernel='rbf')

rbf_svm_grid=GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)

rbf_svm_grid_results=perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)

#observe the attributes of the model

print_grid_search_attributes(rbf_svm_grid_results['model'])


# In[56]:


params = {'n_estimators': np.arange(10,201,20), 'max_depth': np.arange(3,15,2)}

rfc = RandomForestClassifier()

rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)

rfc_grid_results = perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

# observe the attributes of the model

print_grid_search_attributes(rfc_grid_results['model'])


# In[60]:


print('\n                    Accuracy     Error')

print('                      --------     -----')



print('Logistic Regression:  {:.04}%       {:.04}%'.format(log_reg_grid_results['accuracy']*100,\
                                                            100-(log_reg_grid_results['accuracy']*100)))
                                                            

print('Linear SVC         :   {:.04}%       {:.04}%'.format(lr_svc_grid_results['accuracy']*100,\
                                                            100-(lr_svc_grid_results['accuracy']*100)))
                                                            
print('rbf SVM classifier  :   {:.04}%       {:.04}%'.format(rbf_svm_grid_results['accuracy']*100,\
                                                            100-(rbf_svm_grid_results['accuracy']*100)))

print('Random Forest       :   {:.04}%       {:.04}%'.format(rfc_grid_results['accuracy']*100,\
                                                            100-(rfc_grid_results['accuracy']*100)))


# In[ ]:




