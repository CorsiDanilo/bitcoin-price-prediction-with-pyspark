# **Bitcoin price prediction**
### Big Data Computing final project - A.Y. 2022 - 2023
Prof. Gabriele Tolomei

MSc in Computer Science

La Sapienza, University of Rome

### Author
Corsi Danilo - corsi.1742375@studenti.uniroma1.it


# Project details
### Problem
The cryptocurrency Bitcoin has attracted the attention of many people in recent years. However, it's price fluctuation can be extremely unpredictable, which makes it difficult to predict when the right time to buy or sell this digital currency will be. In this context, prediction Bitcoin prices can be a competitive advantage for investors and traders, as it could allow them to make informed decisions on the right time to enter or exit the market. In this project, I will analyze some machine learning techniques to understand, through the processing of historical data, how accurately the price of Bitcoin can be predicted and whether this can provide added value to cryptocurrency investors and traders.
### Dataset ❗
I chose to use the following dataset from Blockchain.org, more specifically those containing minute-by-minute updates of the Bitcoin price from 2017 to 2021 (period for which there were moments of high volatility but also a lot of price lateralisation). The columns (features) contained in it, in addition to the timestamp of each transaction, are the opening, closing, highest and lowest price and the corresponding trading volume in Bitcoin and Dollars.
### Methods ❗
The methods I will test will be Linear Regression (simple and multiple) and Random Forest. Further comparisons with other classification models are planned in the course of development. Moreover, I would also like to try to understand what the differences are between these methods and the implementation of a state-of-the-art neural network such as Long-Short Term Memory.

### Evaluation framework ❗
As evaluation framework I will use R-square (R²), Mean Square Error (MSE) and Mean Absolute Error (MAE) to get a complete picture of the performance of the various models.

# Project structure
```
.
├── datasets
│  │ 
│  ├── output
│  │  ├── bitcoin_blockchain_data_30min_test.parquet
│  │  ├── bitcoin_blockchain_data_30min_train_valid.parquet
│  │ 
│  │── raw
│     ├── bitcoin_blockchain_data_30min.parquet
│  
├── features
│  ├── all_features.json
│  ├── features_correlation.json
│  ├── least_corr_features.json
│  └── most_corr_features.json
│ 
├── models
│  ├── GBTRegressor
│  ├── GeneralizedLinearRegression
│  ├── LinearRegression
│  └── RandomForestRegressor
│ 
├── notebooks
│  ├── 1. Data crawling.ipynb
│  ├── 2. Feature Engineering.ipynb
│  ├── 3. Linear Regression.ipynb
│  ├── 4. Generalized Linear Regression.ipynb
│  ├── 5. Random Forest Regressor.ipynb
│  ├── 6. Gradient Boosting Tree Regressor.ipynb
│  ├── 7. Final predictions.ipynb
│ 
├── results
│  ├── final.csv
│  ├── GBTRegressor.csv
│  ├── GeneralizedLinearRegression.csv
│  ├── LinearRegression.csv
│  └── RandomForestRegressor.csv
│ 
└── utilities
   ├── imports.py
   ├── parameters.py
   └── utilities.py
``` 


# Description
### **Dataset folder:** Contains the original and processed datasets

* **bitcoin_blockchain_data_30min_test.parquet:** Dataset used in the final phase of the project to perform price prediction on never-before-seen data
* **bitcoin_blockchain_data_30min_train_validation.parquet:** Dataset used to train and validate the models used
* **bitcoin_blockchain_data_30min.parquet:** Original dataset obtained by making calls to the Blockchain.com API

### **Features folder:** Contains the features used throughout the project
* **features_correlation.json:** Contains features name and their correlation value 
* **all_features.json:** Contains the name of all features
* **less_rel_features.json:** Contains the name of the features most correlated to the price of Bitcoin
* **more_rel_features.json:** Contains the name of the features least correlated to the price of Bitcoin

### **Models folder:** Contains files related to the trained models with the best parameters, ready to be used to perform price prediction on never-before-seen data

### **Notebooks folder**: Contains notebooks produced
* **1. Data crawling.ipynb:** Data crawling on Bitcoin blochckain by querying Blockchain.com website.
* **2. Feature Engineering.ipynb:** Adding useful features regardings the price of Bitcoin, visualizing data and performing feature selection 
* **3 - 6. <model_name>.ipynb:** Executing the chosen model, first with default values, then choosing the best parameters by performing hyperparameter tuning with cross validation, performance evaluation and training of the final model
* **7. Final predictions.ipynb:** Making predictions on the test set with the final models trained on the whole train/validation set

### **Results folder**: Contains all results obtained from each individual model and final predictions

### **Utilities folder**: Contains files defined by me used by most notebooks to reuse the code
* **imports.py:** Contains imports of external libraries
* **parameters.py:** Contains the parameters used by the models during the train / validation phase
* **utilities.py:** Contains functions that are used by the models during the train / validation phase

# Final results ❗
