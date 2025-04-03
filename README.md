# MSEGNet


Code for the model in the paper MSEGNet: A Hybrid Multi-Scale CNN with Temporal-Contextual Modeling for Enhanced Sleep Staging Using Single-Channel EEG by Dongdong Zhou, Qi Xu, Lauri Kettunen, Qiang Zhang, Fengyu Cong.

This work is under review in [IEEE Transactions on Instrumentation and Measurement](https://ieee-ims.org/publication/ieee-tim).

The structure of MSEGNet is shown as:

![MSEGNet](./images/MSEGNet.png)


## Environment ##

- CUDA toolkit 11.6 and CuDNN v7.6.4
- Python 3.6.7
- tensorflow-gpu (1.12.0)
- Keras (2.2.4)
- matplotlib (3.2.2)
- scikit-learn (0.23.1)
- scipy (1.5.0)
- numpy (1.16.0)
- pandas (1.1.0)
- mne (0.21.2)
- h5py (2.10.0)



## Evaluation datasets ##
We evaluated our MSEGNet with [CCSHS](https://sleepdata.org/datasets/ccshs) and [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) datasets.



## Summary ##
Data preparation

1. The format of data and label are .h5 and .csv, respectively. 
2. Many-to-one scheme: 
![contextual input](./images/contextual input.png)

Run SCNet.py to get the confusion matrix of test set of each fold. 

The mean coufusion matrix of 5-fold cross validation. 

The dimension of the mean confusion matrix is 6*6. The 6th row and column are the sum value.

Run Performance_metrics.py to get the performance metrics: overall accuracy (ACC), precision (PR), recall (RE), F1 score (F1), and Cohen’s kappa coefficient (K), which can be caculated from the mean confusion matrix.

    

## Citation ##

## Licence ##
- For academic and non-commercial use only
- Apache License 2.0
