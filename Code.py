import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm

def class_acc(pred, gt):
    count=0   #number of misclassified samples
    for i_index, i in enumerate(gt):
        if pred[i_index]!= gt[i_index]:
            count= count + 1   
    accuracy= 1 - count/len(gt)
    return accuracy

def cifar10_classifier_random(x):
    random_class= np.random.randint(10, size=50000)
    
    return random_class

def cifar10_classifier_1nn(x,trdata,trlabels):
    
    shortest_distance= []
    label= []
    for i in trdata:
        #Distance= np.sqrt(np.sum((i-x)**2)) #Eucleadian Distance
        Distance= np.sum(np.abs((i-x))) #manhattan distance - preferred due to high dimensionality in data
        shortest_distance.append(Distance)
    
    min_distance= np.argmin(shortest_distance)
    label= trlabels[min_distance]
    return label

def cifar10_classifier(x, trdata, trlabels):
    class_pred=[]
    for i in tqdm(x):
        class_pred.append(cifar10_classifier_1nn(i, trdata, trlabels))
    return class_pred
    

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_1')
X = datadict["data"]
Y = datadict["labels"]

datadict_1 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_2')
X_1 = datadict_1["data"]
Y_1 = datadict_1["labels"]

datadict_2 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_3')
X_2 = datadict_2["data"]
Y_2 = datadict_2["labels"]

datadict_3 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_4')
X_3 = datadict_3["data"]
Y_3 = datadict_3["labels"]

datadict_4 = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/data_batch_5')
X_4 = datadict_4["data"]
Y_4 = datadict_4["labels"]

datadict_tb = unpickle('E:/Study Material/Masters/Studies\Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/test_batch')
X_tb = datadict_tb["data"]
Y_tb = datadict_tb["labels"]

x_merge= np.concatenate([X, X_1, X_2, X_3, X_4])
y_merge= np.concatenate([Y, Y_1, Y_2, Y_3, Y_4])


labeldict = unpickle('E:/Study Material/Masters/Studies/Semester 1/Introduction to Pattern Recognition and Machine Learning - DATA.ML.100/Excercises/Excercise Week 3/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = x_merge.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype('int')
Y = np.array(y_merge)

X_tb= X_tb.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8").astype('int')
Y_tb = np.array(Y_tb)

print(X.shape)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.99999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)


z_1= (class_acc(Y,Y))
print(f'\nAccuracy Function:{round(z_1*100,2)}%')
z_2= (class_acc(cifar10_classifier_random(X),Y))
print(f'\nRandom Classifier:{round(z_2*100,2)}%\n')
z_3= (class_acc(cifar10_classifier(x=X_tb, trdata=X, trlabels=Y), Y_tb))
print(f'\n1NN accuracy:{round(z_3*100,2)}%')
