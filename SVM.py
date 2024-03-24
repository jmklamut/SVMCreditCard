# Importing librairies

import pandas as pd
import numpy as np

# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

# Matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data vizualisation
import seaborn

data = pd.read_csv('./creditcard.csv')  # Reading the file .csv
df = pd.DataFrame(data)  # Converting data to Panda DataFrame

df.describe()  # Description of statistic features (Sum, Average, Variance, minimum, 1st quartile, 2nd quartile, 3rd Quartile and Maximum)
"""
df_fraud = df[df['Class'] == 1] # Recovery of fraud data
plt.figure(figsize=(15,10))
plt.scatter(df_fraud['Time'], df_fraud['Amount']) # Display fraud amounts according to their time
plt.title('Scratter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,175000])
plt.ylim([0,2500])
plt.show()

nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0] # Recovery of frauds over 1000
print('There are only '+ str(nb_big_fraud) + ' frauds where the amount was bigger than 1000 over ' + str(df_fraud.shape[0]) + ' frauds')

number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print('There are only '+ str(number_fraud) + ' frauds in the original dataset, even though there are ' + str(number_no_fraud) +' no frauds in the dataset.')

print("The accuracy of the classifier then would be : "+ str((284315-492)/284315)+ " which is the number of good classification over the number of tuple to classify")

df_corr = df.corr() # Calculation of the correlation coefficients in pairs, with the default method:
                    # Pearson, Standard Correlation Coefficient
plt.figure(figsize=(15,10))
seaborn.heatmap(df_corr, cmap="YlGnBu") # Displaying the Heatmap
seaborn.set(font_scale=2,style='white')

plt.title('Heatmap correlation')
plt.show()
"""
df_corr = df.corr()  # Calculation of the correlation coefficients in pairs, with the default method:
# Pearson, Standard Correlation Coefficient

rank = df_corr['Class']  # Retrieving the correlation coefficients per feature in relation to the feature class
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class', ascending=False)  # Ranking the absolute values of the coefficients
# in descending order
df_rank.dropna(inplace=True)  # Removing Missing Data (not a number)

# We seperate ours data in two groups : a train dataset and a test dataset

# First we build our train dataset
df_train_all = df[0:150000]  # We cut in two the original dataset
df_train_1 = df_train_all[df_train_all['Class'] == 1]  # We seperate the data which are the frauds and the no frauds
df_train_0 = df_train_all[df_train_all['Class'] == 0]


######################## balanced data #######################################################################
df_sample = df_train_0.sample(300)

df_train = df_train_1._append(df_sample)  # We gather the frauds
df_train = df_train.sample(frac=1)  # Then we mix our dataset

X_train = df_train.drop(['Time', 'Class'], axis=1)  # We drop the features Time (useless), and the Class (label)
y_train = df_train['Class']  # We create our label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


####################### unbalanced data #######################################################################

df_sample = df_train_all.sample(2000)

X_train_unbalanced = df_sample.drop(['Time', 'Class'], axis=1)  # We drop the features Time (useless), and the Class (label)
y_train_unbalanced = df_sample['Class']  # We create our label
X_train_unbalanced = np.asarray(X_train_unbalanced)
y_train_unbalanced = np.asarray(y_train_unbalanced)

############################## with all the test dataset to see if the model learn correctly ##################
df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'], axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)

X_train_rank = df_train[df_rank.index[1:11]]  # We take the first ten ranked features
X_train_rank = np.asarray(X_train_rank)


X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

class_names = np.array(['0', '1'])  # Binary label, Class = 1 (fraud) and Class = 0 (no fraud)


# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


##################################Testing balanced and un balanced datasets individually#################
balanced_classifier = svm.SVC(
    kernel='linear')  # We set a SVM classifier, the default SVM Classifier (Kernel = Radial Basis Function)

balanced_classifier.fit(X_train, y_train)  # Then we train our model, with our balanced data train.

unbalanced_classifier = svm.SVC(
    kernel='linear')  # We set a SVM classifier, the default SVM Classifier (Kernel = Radial Basis Function)

unbalanced_classifier.fit(X_train_unbalanced, y_train_unbalanced)  # Then we train our model, with our balanced data train.

balanced_prediction_SVM_all = balanced_classifier.predict(X_test_all)  # And finally, we predict our data test.

unbalanced_prediction_SVM_all = unbalanced_classifier.predict(X_test_all)  # And finally, we predict our data test.

cm = confusion_matrix(y_test_all, balanced_prediction_SVM_all)
plot_confusion_matrix(cm, class_names)

u_cm = confusion_matrix(y_test_all, unbalanced_prediction_SVM_all)
plot_confusion_matrix(u_cm, class_names)

print('Our criterion for the balanced data is '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

print('Our criterion for the unbalanced data is '
      + str(((u_cm[0][0] + u_cm[1][1]) / (sum(u_cm[0]) + sum(u_cm[1])) + 4 * u_cm[1][1] / (u_cm[1][0] + u_cm[1][1])) / 5))

print('We have detected ' + str(u_cm[1][1]) + ' frauds / ' + str(u_cm[1][1] + u_cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(u_cm[1][1] / (u_cm[1][1] + u_cm[1][0])))
print("the accuracy is : " + str((u_cm[0][0] + u_cm[1][1]) / (sum(u_cm[0]) + sum(u_cm[1]))))


#########################################Testing unbalanced and balanced datasets together###################

balanced_classifier.fit(X_train_unbalanced, y_train_unbalanced)

balanced_prediction_SVM_all = balanced_classifier.predict(X_test_all)  # And finally, we predict our data test.

cm = confusion_matrix(y_test_all, balanced_prediction_SVM_all)
plot_confusion_matrix(cm, class_names)

print('Our criterion for the balanced/unbalanced data is '
      + str(((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))

print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1] / (cm[1][1] + cm[1][0])))
print("the accuracy is : " + str((cm[0][0] + cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


