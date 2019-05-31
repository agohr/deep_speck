import speck as sp
import numpy as np

from keras.models import model_from_json

#load distinguishers
json_file = open('single_block_resnet.json','r');
json_model = json_file.read();

net5 = model_from_json(json_model);
net6 = model_from_json(json_model);
net7 = model_from_json(json_model);
net8 = model_from_json(json_model);

net5.load_weights('net5_small.h5');
net6.load_weights('net6_small.h5');
net7.load_weights('net7_small.h5');
net8.load_weights('net8_small.h5');

def evaluate(net,X,Y):
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

X5,Y5 = sp.make_train_data(10**6,5);
X6,Y6 = sp.make_train_data(10**6,6);
X7,Y7 = sp.make_train_data(10**6,7);
X8,Y8 = sp.make_train_data(10**6,8);

X5r, Y5r = sp.real_differences_data(10**6,5);
X6r, Y6r = sp.real_differences_data(10**6,6);
X7r, Y7r = sp.real_differences_data(10**6,7);
X8r, Y8r = sp.real_differences_data(10**6,8);

print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
print('5 rounds:');
evaluate(net5, X5, Y5);
print('6 rounds:');
evaluate(net6, X6, Y6);
print('7 rounds:');
evaluate(net7, X7, Y7);
print('8 rounds:');
evaluate(net8, X8, Y8);

print('\nTesting real differences setting now.');
print('5 rounds:');
evaluate(net5, X5r, Y5r);
print('6 rounds:');
evaluate(net6, X6r, Y6r);
print('7 rounds:');
evaluate(net7, X7r, Y7r);
print('8 rounds:');
evaluate(net8, X8r, Y8r);
