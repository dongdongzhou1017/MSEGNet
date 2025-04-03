# MSEGNet


Code for the model in the paper MSEGNet: A Hybrid Multi-Scale CNN with Temporal-Contextual Modeling for Enhanced Sleep Staging Using Single-Channel EEG by Dongdong Zhou, Qi Xu, Lauri Kettunen, Qiang Zhang, Fengyu Cong.

This work is under review in [IEEE Transactions on Instrumentation and Measurement](https://ieee-ims.org/publication/ieee-tim).

The structure of MSEGNet is shown as:

![MSEGNet](./images/MSEGNet.png)


## Environment ##

- CUDA toolkit 12.6 and CuDNN v8.1.0
- Python 3.10.9
- tensorflow-gpu (2.10.0)
- matplotlib (3.7.0)
- scikit-learn (1.5.0)
- scipy (1.10.1)
- numpy (1.23.5)
- pandas (1.5.3)
- mne (1.5.0)
- h5py (3.7.0)



## Evaluation datasets ##
We evaluated our MSEGNet with [CCSHS](https://sleepdata.org/datasets/ccshs) and [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) datasets.



## Summary ##
Data preparation

1. The format of data and label are .h5 and .csv, respectively. 
2. Construction of Different EEG Input Lengths: 
![contextual input](./images/contextual-input.png)
The construction of inputs involves varying lengths, denoted by $\mathbf{X}_{m}$, $\mathbf{P}_{m}$, \textcolor{red}{$\mathbf{S}_{m}$},  \textcolor{red}{$\mathbf{C9}_{m}$},
 and  \textcolor{red}{$\mathbf{C15}_{m}$}, corresponding to the conventional 30s input, preceding 60s input, succeeding 60s input, contextual 90s input,
 and contextual 150s input, respectively. $\mathbf{P}_{m}$ contains $\mathbf{X}_{m-1}$, $\mathbf{X}_{m}$, \textcolor{red}{$\mathbf{S}_{m}$} consists of
 $\mathbf{X}_{m}$, $\mathbf{X}_{m+1}$,  \textcolor{red}{$\mathbf{C9}_{m}$} comprises $\mathbf{X}_{m-1}$, $\mathbf{X}_{m}$ and $\mathbf{X}_{m+1}$,  \textcolor{red}{$\mathbf{C15}_{m}$} is
 comprised of $\mathbf{X}_{m-2}$, $\mathbf{X}_{m-1}$, $\mathbf{X}_{m}$, $\mathbf{X}_{m+1}$ and $\mathbf{X}_{m+2}$.  \textcolor{red}{The labels for the different input lengths corresponded to the subscript notation, where the labels for $\mathbf{P}_{m}$, $\mathbf{S}_{m}$,  \textcolor{red}{$\mathbf{C9}_{m}$}, and  \textcolor{red}{$\mathbf{C15}_{m}$} are denoted as $y_{m}$.}

Run construction_model_input.m to construct preceding 60s, succeeding 60s, contextual 90s, and contetual 90s input.

Run MSEGNet.py to get the confusion matrix of test set of each fold. 

The mean coufusion matrix of 5-fold cross validation. 

The dimension of the mean confusion matrix is 6*6. The 6th row and column are the sum value.

Run Performance_metrics.py to get the performance metrics: overall accuracy (ACC), precision (PR), recall (RE), F1 score (F1), and Cohen’s kappa coefficient (K), which can be caculated from the mean confusion matrix.

    

## Citation ##

## Licence ##
- For academic and non-commercial use only
- Apache License 2.0
