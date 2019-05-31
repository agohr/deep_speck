#test if rapid training runs using neural networks can be used to find good initial differences in Speck32/64
#idea: try all differences up to a certain weight and keep track of the performance level reached

import train_nets as tn
import speck as sp
import numpy as np

from keras.models import Model
from sklearn.linear_model import Ridge

from random import sample, randint
from collections import defaultdict
from math import log2

linear_model = Ridge(alpha=0.01);

#first, we train a network to distinguish 3-round Speck with a randomly chosen input difference
#then, we use the penultimate layer output of that network to preprocess output data for other differences
#The preprocessed output for 1000 examples each is fed as training data to a single-layer perceptron
#That single-layer perceptron is then evaluated using 1000 validation samples
#The validation accuracy of the perceptron is taken as an indication of how much the differential distribution deviates from uniformity for that input difference

def train_preprocessor(n, nr, epochs):
  net = tn.make_resnet(depth=1);
  net.compile(optimizer='adam',loss='mse',metrics=['acc']);
  #create a random input difference
  diff_in = (randint(0,2**16), randint(0,2**16));
  X,Y = sp.make_train_data(n, nr, diff=diff_in);
  net.fit(X,Y,epochs=epochs, batch_size=5000,validation_split=0.1);
  net_pp = Model(inputs=net.layers[0].input, outputs=net.layers[-2].output);
  return(net_pp);

def evaluate_diff(diff, net_pp, nr=3, n=1000):
  if (diff == 0): return(0.0);
  d = (diff >> 16, diff & 0xffff);
  X,Y = sp.make_train_data(2*n, nr,diff=d);
  Z = net_pp.predict(X,batch_size=5000);
  #perceptron.fit(Z[0:n],Y[0:n]);
  linear_model.fit(Z[0:n],Y[0:n]);
  #val_acc = perceptron.score(Z[n:],Y[n:]);
  Y2 = linear_model.predict(Z[n:]);
  Y2bin = (Y2 > 0.5);
  val_acc = float(np.sum(Y2bin == Y[n:])) / n;
  return(val_acc);

#for a given difference, derive a guess how many rounds may be attackable
def extend_attack(diff, net_pp, nr, val_acc):
  print("Estimates of attack accuracy:");
  while(val_acc > 0.52):
    print(str(nr) + " rounds:" + str(val_acc));
    nr = nr + 1;
    val_acc = evaluate_diff(diff, net_pp, nr=nr, n=1000);

def greedy_optimizer_with_exploration(guess, f, n=2000, alpha=0.01, num_bits=32):
  best_guess = guess;
  best_val = f(guess); val = best_val;
  d = defaultdict(int)
  for i in range(n):
    d[guess] = d[guess] + 1;
    r = randint(0, num_bits-1);
    guess_neu = guess ^ (1 << r);
    val_neu = f(guess_neu);
    if (val_neu > best_val):
      best_val = val_neu; best_guess = guess_neu;
      print(hex(best_guess), best_val);
    if (val_neu - alpha*log2(d[guess_neu]+1) > val - alpha*log2(d[guess]+1)):
      val = val_neu; guess = guess_neu;
  return(best_guess, best_val);

net_pp = train_preprocessor(10**7,3,1);
for i in range(10):
  print("Run ",i,": ");
  diff, val_acc = greedy_optimizer_with_exploration(randint(0,2**32-1), lambda x: evaluate_diff(x, net_pp, 3));
  extend_attack(diff, net_pp, 3, val_acc);
