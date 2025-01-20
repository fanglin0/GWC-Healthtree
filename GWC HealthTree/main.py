import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "GenHealth","Smoking", "AlcoholDrinking", "PhysicalActivity", "AgeCategory", "Sex"])

df = util.oneHotEncoder(df, ["Race"])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset


print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)


input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
X = df.drop("HeartDisease", axis= 1)
y = df["HeartDisease"]

random_state = 3
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 7, class_weight="balanced")
clf = clf.fit(X_train, y_train)

#Test the model with the testing data set and prints accuracy score
test_prediction = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_prediction)

print("The accuracy with the testing data set of the Decision Tree is:" +str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_prediction, labels = [1, 0])
print("The confusion matrix of the tree is: ")
print(cm)

#Test the model with the training data set and prints accuracy score
train_prediction = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_prediction)

print("The accuracy with the training data set of the Decision Tree is:" +str(train_acc))



input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations






#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)
print("\nI found the commonality that most patient classfied as having Heart Disease usually fell into the BMI [0.00, 5.91] category.\n")

#Prints how a Decision Tree can be used in another field
print("\nSomething I'm interested in is media, from books to movies and anything that involves storytelling. I feel like these topics can use Decision Trees to aide with deciding and recommending similar media, based on whether you liked your original media or not. \nOr it can be used to predict whether someone would enjoy a piece of media using current knowledge on which pieces they enjoyed and which ones not.\n")
print("\nWhen creating the Decision Tree, some factors to be careful of is biases present in training data, that can skew the final results present.as well as to include a confusion matrix to ensure accuracy despite what the accuracy score may be.\n Also, remembering to compare testing and training acuracy scores is important as well.\n")