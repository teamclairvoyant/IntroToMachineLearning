'''
This simple code is desinged to teach how to use python/ ipython notebook to read datafiles, 
find what proportion of males and females survived in titanic crash 
and make a DecisionTreeClassifier predictive model using Scikit Learn
Author : SatyaMudiam
Date : 26 February 2015

'''
# import required python packages. This can be done in the middle of the code as well just before calling the functions of the package.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder

titanic = pd.read_csv("** path/train.csv on your machine **")
# print titanic dataframe to understand the data. 
print titanic
# print all columns in the data 
print titanic.columns.values
# describe numeric columns in data
print titanic.describe().columns.values

# From data we are selecting pclass,age,Gender for training and survived is our taget we are trying to predict.
# This is called Feature Selection
dataColumns = [ 'Pclass', 'Age', 'Gender','Survived']
titanicdata = titanic[dataColumns]

print titanicdata
# shape return dimensions of the dataframe/array
print titanicdata.shape

# print size of Array that holds [Age] data
print titanicdata['Age'].size

# Check if there are any NA or NaN (unknown) values in numerical fields. If we have categorical data, we have to convert it to numerical first and then check. We will do this step with [Gender] field.
# If the result is True we have to address the case of missing values. In our scenario, we have missing values for Age. 
print np.isnan(titanicdata['Age']).any()
print np.isnan(titanicdata['Pclass']).any()

# Find average of [Age] and fill in NaN values with the average. 
# To do that, first fill in NaNs with zeros. We can not compute mean with NaN in dataset. 
titanicdata['Age'].fillna(0, inplace=True)
print titanicdata

# Exclude entries that have Age as zero in computing mean. There may be other better ways to handle this scenario.
avg_age = np.mean(titanicdata[titanicdata['Age'] != 0])['Age']
print avg_age
# Replace zeros with avg_age we just computed
titanicdata['Age'] = titanicdata['Age'].apply(lambda x: avg_age if x == 0 else x)
print titanicdata

# Now convert categorical values of Gender array into Numerical values that start from 0. DecisionTree Calssifier requires the fields to be in numerical form.
# use LabelEncoder and train it wih Gender array to identify all the categories present. 
# This is called Feature Extraction
enc = LabelEncoder()
label_encoder = enc.fit(titanicdata['Gender'])
print "Categorical Classes:", label_encoder.classes_ 

# Tranform categorical values into numerical values and print them
integer_classes = label_encoder.transform(label_encoder.classes_)
print "Integer Classes:", integer_classes

# Now transform [Gender] array's categorical values into numerical values.
titanicdata['Gender']  = label_encoder.transform(titanicdata['Gender'])
# check if we have NaNs in [Gender] field
print np.isnan(titanicdata['Gender']).any()

# We should see all numerical fields in our training data.
print titanicdata

# Ltes create titanic_x and titanic_y for training fields and for target ['survived'].
titanic_x, titanic_y = titanicdata[['Pclass', 'Age', 'Gender']] , titanicdata['Survived']
# print first entry
print titanic_x[0] , titanic_y[0]

# We need to split initial train.csv(training data) into 2 splits. Use one split for training the model and the other split to test the model. We split it by 80:20 for training and testing.
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(titanic_x, titanic_y, test_size=0.80, random_state=33)
# print x_test
# print y_test

# Using DecisionTreeClassifier
# Selecting model parameters is called Model Selection
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
# Train the model using training data (80% of train.csv)
clf = clf.fit(x_train, y_train)

# If you want to plot out Decision nodes in the tree, uncomment below code and make sure you have pydot and pyparsing packes

# import pydot, StringIO
# dot_data = StringIO.StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=['Pclass', 'Age', 'Gender'])
# print dot_data.getvalue()
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('titanic_png')

# from IPython.core.display import Image
# Image(filename='titanic_png')

from sklearn import metrics
# Predict the outcome [Survived]  for training data. Model may perform well on training data as it has seen it already.
# This is just for understanding purposes. We may never measure accuracy or rest of the mtrics on training data itself.
y_train_pred = clf.predict(x_train)
print "Training Accuracy:" , metrics.accuracy_score(y_train, y_train_pred)

# Now predict for the test data we saved from train.csv (20% of the train.csv data)
y_test_pred = clf.predict(x_test)
# Print Accuracy by comparing actual ['Survived'] data we have from train.csv and the prediction from the model.
print "Test Accuracy:" , metrics.accuracy_score(y_test, y_test_pred)

# Print Confusion Matrix by comparing Actual to predicted outcomes. TruePositive and TrueNegative values should be high the matrix.
print "Confusion Matrix"
print metrics.confusion_matrix(y_test, y_test_pred)

# Classification Report shows Precission and recall that are used to measure a model's performance. 
print "Classification Report:"
print metrics.classification_report(y_test, y_test_pred)


# Now lets actually predict for the test.csv, where we do not know/ have the outcome [Survived]
# We follow same steps we followed with training data on feature selection and extraction.
test_df = pd.read_csv("** path/test.csv on your local machine **")
# print testdf

test = test_df[['Pclass', 'Age', 'Gender']]
print test.head(5)

# print test['Age'].shape
# print test['Pclass'].size

#data1 = test_df.ix[10:15, ['Pclass', 'Age', 'Gender']]
# print data1.head(5)
test['Age'].fillna(0, inplace=True)
test_avg_age = np.mean(test[test['Age'] !=0 ])['Age']
print test_avg_age

test['Age'] = test['Age'].apply(lambda x: test_avg_age if x == 0 else x)

print test['Age'].head(4)

test['Gender'] = label_encoder.transform(test['Gender'])
x_test_data = test

# Predict for test data from test.csv
y_test_data = clf.predict(x_test_data)
print y_test_data[2]

# Append our prediction to original test data
test_df['Survived'] = y_test_data
print test_df.head(10)

# Save test data and prediction data into a csv on your machine. 
test_df.to_csv("** provide the location of the folder**/test_with_predictions.csv", index=False, index_label=None)

# At the end you should see test_with_predictions.csv file created with test.csv data and its predictions associated.




