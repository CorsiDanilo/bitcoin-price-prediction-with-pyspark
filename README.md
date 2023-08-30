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

### Dataset 
I chose to collect data on the Bitcoin blockchain using the API of the website Blockchain.org, the most relevant information was retrieved from the year 2016 to the present day (a period for which there were moments of high volatility but also a lot of price lateralization). 
The features taken under consideration were divided into several categories:

**Currency Statistics**
* **market-price:** Market Price: The average USD market price across major bitcoin exchanges.
* **trade-volume:** Exchange Trade Volume (USD): The total USD value of trading volume on major bitcoin exchanges.
* **total-bitcoins:** Total Circulating Bitcoin: The total number of mined bitcoin that are currently circulating on the network.
* **market-cap:** Market Capitalization (USD): The total USD value of bitcoin in circulation.

**Block Details**
* **blocks-size:** Blockchain Size (MB): The total size of the blockchain minus database indexes in megabytes.
* **avg-block-size:** Average Block Size (MB): The average block size over the past 24 hours in megabytes.
* **n-transactions-total:** Total Number of Transactions: The total number of transactions on the blockchain.
* **n-transactions-per-block:** Average Transactions Per Block: The average number of transactions per block over the past 24 hours.

**Mining Information**
* **hash-rate:** Total Hash Rate (TH/s): The estimated number of terahashes per second the bitcoin network is performing in the last 24 hours.
* **difficulty:** Network Difficulty (T): A relative measure of how difficult it is to mine a new block for the blockchain.
* **miners-revenue:** Miners Revenue (USD): Total value of coinbase block rewards and transaction fees paid to miners.
* **transaction-fees-usd:** Total Transaction Fees (USD): The total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.

**Network Activity**
* **n-unique-addresses:** The total number of unique addresses used on the blockchain.
* **n-transactions:** Confirmed Transactions Per Day: The total number of confirmed transactions per day.
* **estimated-transaction-volume-usd:** Estimated Transaction Value (USD): The total estimated value in USD of transactions on the blockchain.

### Methods 
The methods I will test will be **Linear Regression**, **Generalized Linear Regression**, **Random Forest Regressor** and **Gradient Boosting Tree Regressor** in order to see their differences and how they perform through various stages of train/validation and testing.

### Evaluation framework 

As evaluation framework I will use **RMSE (Root Mean Squared Error)**, **MSE (Mean Squared Error)**, **MAE (Mean Absolute Error)**, **MAPE (Mean Absolute Percentage Error)**, **R2 (R-squared)** and **Adjusted R2** to get a complete picture of the performance of the various models.

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
