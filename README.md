# ****Bitcoin price prediction****

### **Big Data Computing final project - A.Y. 2022 - 2023**

MSc in Computer Science

La Sapienza, University of Rome

# **Project details**

### **Problem**

The cryptocurrency Bitcoin has attracted the attention of many people in recent years. However, it's price fluctuation can be extremely unpredictable, which makes it difficult to predict when the right time to buy or sell this digital currency will be. In this context, prediction Bitcoin prices can be a competitive advantage for investors and traders, as it could allow them to make informed decisions on the right time to enter or exit the market. In this project, I will analyze some machine learning techniques to understand, through the processing of historical data, how accurately the price of Bitcoin can be predicted and whether this can provide added value to cryptocurrency investors and traders.

### **Dataset**

I chose to collect data on the Bitcoin blockchain using the API of the website Blockchain.org, the most relevant information was retrieved from the year 2016 to the present day (a period for which there were moments of high volatility but also a lot of price lateralization). The features taken under consideration were divided into several categories:

**Currency Statistics**

- **market-price:** the average USD market price across major bitcoin exchanges.
- **trade-volume:** the total USD value of trading volume on major bitcoin exchanges.
- **total-bitcoins:** the total number of mined bitcoin that are currently circulating on the network.
- **market-cap:** the total USD value of bitcoin in circulation.

**Block Details**

- **blocks-size:** the total size of the blockchain minus database indexes in megabytes.
- **avg-block-size:** the average block size over the past 24 hours in megabytes.
- **n-transactions-total:** the total number of transactions on the blockchain.
- **n-transactions-per-block:** the average number of transactions per block over the past 24 hours.

**Mining Information**

- **hash-rate:** the estimated number of terahashes per second the bitcoin network is performing in the last 24 hours.
- **difficulty:** a relative measure of how difficult it is to mine a new block for the blockchain.
- **miners-revenue:** total value of coinbase block rewards and transaction fees paid to miners.
- **transaction-fees-usd:** the total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.

**Network Activity**

- **n-unique-addresses:** the total number of unique addresses used on the blockchain.
- **n-transactions:** the total number of confirmed transactions per day.
- **estimated-transaction-volume-usd:** the total estimated value in USD of transactions on the blockchain.

### **Methods**

The methods I will test will be **Linear Regression**, **Generalized Linear Regression**, **Random Forest Regressor** and **Gradient Boosting Tree Regressor** in order to see their differences and how they perform through various stages of train/validation and testing.

### **Evaluation framework**

As evaluation framework I will use **RMSE (Root Mean Squared Error)**, **MSE (Mean Squared Error)**, **MAE (Mean Absolute Error)**, **MAPE (Mean Absolute Percentage Error)**, **R2 (R-squared)** and **Adjusted R2** to get a complete picture of the performance of the various models.

# **Project structure**

```
.
├── datasets
│  │
│  ├── output
│  │  ├── bitcoin_blockchain_data_15min_test.parquet
│  │  ├── bitcoin_blockchain_data_15min_train_valid.parquet
│  │
│  │── raw
│     ├── bitcoin_blockchain_data_15min.parquet
│
├── features
│  ├── all_features.json
│  ├── features_relevance.json
│  ├── least_rel_features.json
│  └── most_rel_features.json
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
|  ├── block_splits
|  ├── walk_forward_splits
|  ├── short_term_split
│
│
└── utilities
   ├── imports.py
   ├── parameters.py
   └── utilities.py
```

# Files / folder d**escription**

### **Datasets folder: contains the original and processed datasets**

- **bitcoin_blockchain_data_15min_test.parquet:** dataset used in the final phase of the project to perform price prediction on never-before-seen data
- **bitcoin_blockchain_data_15min_train_validation.parquet:** dataset used to train and validate the models used
- **bitcoin_blockchain_data_15min.parquet:** original dataset obtained by making calls to the Blockchain.com API

### **Features folder: contains the features used throughout the project**

- **features_relevance.json:** contains features name and their relevance value
- **all_features.json:** contains the name of all features
- **less_rel_features.json:** contains the name of the most relevant features most with respect to the price of Bitcoin
- **more_rel_features.json:** contains the name of the least relevant features most with respect to the price of Bitcoin

### **Models folder: contains files related to the trained models with the best parameters, ready to be used to perform price prediction on never-before-seen data**

### **Notebooks folder: contains notebooks produced**

- **1. Data crawling.ipynb:** crawling data on bitcoin's blochckain by querying blockchain.com
- **2. Feature Engineering.ipynb:** adding useful features regardings the price of Bitcoin, visualizing data and performing feature selection
- **3 - 6. [<splitting_method>] <model_name>.ipynb:** executing the chosen model, first with default values, then by choosing the best parameters by performing hyperparameter tuning with cross validation and performance evaluation
- **7. Final scores.ipynb:** display of final scores andmaking predictions on the test set with the models trained on the whole train / validation set

### **Results folder: contains all results obtained from each individual model based on the different splittng method**

### **Utilities folder: contains files defined by me used by most notebooks to reuse the code**

- **imports.py:** contains imports of external libraries
- **parameters.py:** contains the parameters used by the models during the train / validation phase
- **utilities.py:** contains functions that are used by the models during the train / validation phase

# Final results ❗