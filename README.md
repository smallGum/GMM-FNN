# Fitting Imbalanced Uncertainties in Multi-Output Time Series Forecasting

This repository releases the code and data for the proposed GMM-FNN model in our paper [Fitting Imbalanced Uncertainties in Multi-Output Time Series Forecasting](https://dl.acm.org/doi/10.1145/3584704).



## Dependencies

+ Python 3.7.6
+ Numpy 1.20.3
+ Pytorch 1.8.1



## Directories

+ `data`: dataset including the three real-world datasets used in our paper.
+ `models`: python code for building the GMM-FNN model.
+ `utils`: python code for auxiliary training.
+ `experiments`: python code for training and testing the GMM-FNN model.
+ `paper_results`: the prediction results of GMM-FNN on the three real-world datasets shown in our paper.



## Usage

To run GMM-FNN, users need to specify the name of the time series dataset they wish to evaluate, the historical (input) length and the prediction (output) length of the time series. Then you can run the script `main_GMMFNN.py`. There are 3 datasets that users can evaluate: ETTh1, WTH and NASDAQ. The script will then evaluate GMM-FNN on the chosen dataset with the same experiment setup presented in our paper. For example, if users want to evaluate GMM-FNN with the system x264, the command line to run GMM-FNN will be:

```shell
$ python main_GMMFNN.py ETTh1 168 24
```

When finishing training and testing,  the script will output the experiment results in the `results` directory.



Also, you can obtain the prediction results of our paper by running the script `paper_results.py`:

```shell
$ python paper_results.py
```

