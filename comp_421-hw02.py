# coding: utf-8
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Reading the image data into a list of lists.
with open("hw02_images.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(list(cont) for cont in csv.reader(f, delimiter=','))
    f.close()
data = [[float(x) for x  in sublist] for sublist in data]    
# Reading the labels into a list.
labels = list(csv.reader(open("hw02_labels.csv", "r"), delimiter=","))
# Splitting the read data into two arrays, namely test_data and train_data.
train_data = data[:len(data)//2]
test_data = data[len(data)//2:]
# Defining constants.
eta = 0.0001
epsilon = 0.001
max_iteration = 500
# Reading W and w0 values and storing them.
with open("initial_W.csv", 'r') as f:
    reader = csv.reader(f)
    W = list(list(float(x) for x in cont) for cont in csv.reader(f, delimiter=','))
    f.close()   
w0 = list(csv.reader(open("initial_w0.csv", "r"), delimiter=","))
# Editing W and w0 values.
for i in range(len(w0)):
    w0[i] = float(w0[i][0])   
WT = list(map(list, zip(*W)))
# Saving each unique label in an array
unique_labels = np.unique(labels)
# Generating Y_truth matrix
y_truth = list()
for i in range(len(unique_labels)):
    y_truth.append(list())
    for j in range(len(labels)):
        y_truth[i].append(1 if (int(labels[j][0]) == i+1) else 0)
# Saving the first half for training.        
Y_train = np.transpose(y_truth)[:len(y_truth[0])//2]
# Saving the other half for testing.
Y_test = np.transpose(y_truth)[len(y_truth[0])//2:]
# Defining the sigmoid function.
def sigmoid(x):
    return (1/(1+np.exp(-x)))
# Defining the gradient function for W.
def gradient_W(X,Y_truth,Y_pred):
    return -(Y_truth - Y_pred) * Y_pred * (1-Y_pred) @ X
# Defining the gradient function for w0.
def gradient_w0(Y_truth,Y_pred):
    return -sum(np.transpose((Y_truth - Y_pred) * Y_pred * (1-Y_pred)))
    # Defining a function to perform the regression, make predictions, and calculate errors.
def reg(X,Y_truth,W,w0):
    iter = 0
    obj_vals = list()
    while iter <= max_iteration:
        # Saving the predicted values.
        Y_pred = sigmoid(np.column_stack((W,w0)) 
                         @ np.transpose(np.column_stack((X,np.ones(len(X))))))
        # Calculating delta_W and delta_w0, and then updating W and w0.
        delta_W = -eta*gradient_W(X,np.transpose(Y_truth),Y_pred)
        W += delta_W
        delta_w0 = -eta*gradient_w0(np.transpose(Y_truth),Y_pred)
        w0 += delta_w0
        obj_vals.append(sum(sum((Y_truth - np.transpose(Y_pred))**2))/2)
        # Calculating errors.
        if(np.sqrt(sum(delta_w0**2) + sum(sum(delta_W**2))) < epsilon):
            break
        iter+=1
    return Y_pred, obj_vals, iter, W, w0
# Saving calculations for both the train and test sets.
Y_pred_train, obj_vals_train, iter_tr, WT, w0 = reg(train_data, Y_train, WT, w0)
Y_pred_test = sigmoid(np.column_stack((WT,w0))
                      @ np.transpose(np.column_stack((test_data,np.ones(len(test_data))))))
# Plotting the error function for the train set.
# %matplotlib inline (for running on a jupyter notebook)
plt.plot(range(iter_tr),obj_vals_train)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show(block = False)
# Finalizing prediction lists naming labels based on the resulting predictions.
def fin_pred(Y_pred):
    yp = list()
    rows, cols = np.shape(Y_pred)
    for i in range(rows):
        yp.append(np.argmax(Y_pred[i])+1)
    return yp

yp_tr = fin_pred(np.transpose(Y_pred_train))
yp_te = fin_pred(np.transpose(Y_pred_test))
# Editing labels list.
y_init = list()
for i in range (int(len(labels))):
    y_init.append(int(labels[i][0]))
# Printing confusion matrix for the test data set.
print(pd.crosstab(np.array(yp_tr), np.array(y_init[:len(labels)//2]), rownames = ['y_pred'], colnames = ['y_train']).head())
# Printing confusion matrix for the train data set.
print(pd.crosstab(np.array(yp_te), np.array(y_init[len(labels)//2:]), rownames = ['y_pred'], colnames = ['y_test']).head())
# To keep the plot showing:
plt.show()
