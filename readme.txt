A) Summary

This repository holds supplementary code and data to the paper "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning". In particular, it contains:

- a proof of concept implementation of the 11-round attack on Speck32/64 (test_key_recovery.py), with parameters for a practical 12-round attack at the very end of the code,
- a basic training script that will train a 10-block deep residual network to distinguish 5-round Speck from random data using the basic training pipeline described in the paper (train_5_rounds.py),
- pre-trained neural distinguishers for 5,6,7 and 8 rounds of Speck,
- a proof of concept implementation of neural input difference search using few-shot-learning (neural_difference_search.py),
- a script that evaluates the pre-trained neural distinguishers both in the real-vs-random setting and in the real differences setting of section 5 of the paper (eval.py),
- a script (key_rank.py) that calculates key rank statistics for the 9-round attack of the paper with a neural distinguisher when between 32 and 128 ciphertext pairs of data are available,
- another script (key_rank_ddt.py) that implements the same experiment using DDT-based distinguishers,
- a python program that looks at whether the performance of the 7-round neural distinguisher supplied in this repository is key-dependent (key_nonuniformity.py),
- a python script that implements the KeyAveraging algorithm from the paper,
- a program written in C++ (speck.cpp in the cpp subfolder) that allows to check with a modest computational budget (less than a day on a standard PC) various claims of the paper about the difference distribution of five-round Speck, the power of key-search based distinguishers, the probability of the most likely differential transition for 8-round Speck, and properties of specific output pairs found in the paper,
- a program written in C++ (speck_ddt.cpp in the cpp subfolder) that calculates the difference distribution for up to 8 rounds of Speck32/64 given the input difference 0x0040/0000 and writes the results to disk. This requires a machine with about 70 GB of unused RAM and about 300 CPU days calculation time. It will compute the predicted output distributions for the output of each round consecutively and write the result to disk; these files are about 35 GB each in size. The resulting files can be loaded into a Python environment by calling numpy.fromfile(filename) and the arrays so generated can be used by key_rank_ddt.py to run the 9-round attack using the difference distribution table.

This archive also contains precomputed key-rank data for the 9-round attack (subdirectory data_9r_attack), data on the key dependence of the 7-round distinguisher, the complete learning history of our best 5-round neural distinguisher (as referenced in the paper) and the wrong key response profiles for all of the supplied pre-trained neural networks except the 5-round one. The learning history of the 5-round network described in the paper is stored as a pickled dictionary in the supplementary data subdirectory.

B) Dependencies and system requirements

Requirements for running the main attack: python3, current keras installation, h5py. Tested with the tensorflow backend, but the code should be backend-agnostic. Training and evaluation code have the same requirements. Tested configuration: keras 2.1.5, tensorflow 1.6.0, h5py 2.7.1.

The neural difference search demonstrator additionally needs a current version of scikit-learn to be installed. Tested version: scikit-learn 0.19.1.

The C++ programs should work with any version of g++ that supports C++2014. g++7.3 has been tested. The code uses gcc builtin functions, so other compilers will not work without changes to the code.

C) Compiling the cpp-files

To build the C++ programs, run make in the cpp subdirectory. This will produce two executable files:

- speck_analysis: runs a number of experiments that will produce evidence of correctness of various claims in the paper (computational budget around one day on a standard PC).

- speck_ddt: calculates the difference distribution of Speck with the input difference used in the paper. This takes significant memory, hard drive space and computing power.

All programs in this code repository (with the possible exception of the key rank and DDT calculation scripts) should work reasonably well on a standard PC. In particular the 11-round attack code should be quite fast without GPU support.

D) Running the experiments

Each of the python files mentioned contains a script that will run a particular set of experiments when run from the terminal using python3.

Instructions for each of the experiments:

1. test_key_recovery.py: run from terminal using the command python3 test_key_recovery.py .

Note that this will run the 11-round attack 100 times and then run a 12-round attack (not described in the paper but using exactly the same code) 20 times. The latter will take some time unless a fast GPU is available (one run of the 12-round attack should take roughly 12 hours on a quad-core PC without GPU). Note that the 12-round attack is with these settings successful only in about 40 percent of cases, so one run is likely not going to be sufficient to demonstrate that it works. Remove the last few lines of the script to turn either behaviour off.

The 11-round attack will produce three files that contain numpy arrays with the following information:

- run_sols1.npy contains the bit difference between real subkey and best guess for the last subkey for each run.
- run_sols2.npy contains the bit difference between real subkey and best guess for the penultimate subkey for each run.
- run_good.npy records the maximal number of good transitions of the initial differential for all ciphertext structures used in each run. This information is given to show which runs were solvable in principle using the techniques described in the paper.

During the run, the script will show how many test cases have been completed. For each completed test, also the bit difference between the final guesses of the last two subkeys and the actual last subkeys is shown.

At the end of the run, information on average attack runtime and data usage is printed.

2. eval.py: run from terminal using python3 eval.py. The results should be self-explanatory.

3. train_5_rounds.py: run from terminal using python3 train_5_rounds.py. Data will be written to the ./freshly_trained_networks/ subdirectory.

4. speck.cpp: compile as directed above and run the resulting speck_analysis binary from terminal. Output should be self-explanatory. The csv file produced can be read by the readcsv function of the python module implementing speck. It is then possible to compare the predictions made by the Markov model to predictions produced for instance by the pre-trained five-round neural network here included.

5. neural_difference_search.py: run from terminal using python3 neural_difference_search.py. This will first briefly train a fresh 3-round distinguisher for Speck with a random input difference and then use this distinguisher with few-shot-learning and the generic optimization algorithm described in the paper to search for input differences that can be more efficiently distinguished. Progress (few-shot distinguisher efficiency for three rounds, extensions to more rounds, input difference) is shown each time an improvement is found. The search is restarted ten times to give a sample of possible results of the algorithm.

6. key_rank.py: generates key-rank statistics for a simple 9-round attack on Speck32/64 when the attack is using a neural distinguisher.

7. key_rank_ddt.py: use the stats_key_rank function to do the same as the previous script, but using a difference distribution table. The DDT needs to be separately loaded. 

8. key_averaging.py: implements KeyAveraging and the creation of high-quality training data using the output of KeyAveraging.

E) Final remarks

The pre-trained networks included in this directory are all small networks with just one residual block. In the five and six round cases, their predictive performance is therefore slightly lower than claimed in the paper. Run the five-round training script to obtain a five-round distinguisher with the performance given in the paper.

F) Citing

If you use the code in this repository for your own research and publish about it, please cite the paper:

Aron Gohr, Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning, Advances in Cryptology - CRYPTO 2019 (to appear)

