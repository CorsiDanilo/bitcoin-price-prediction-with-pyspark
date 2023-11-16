# **Bitcoin price prediction with PySpark**
## Big Data Computing final project - A.Y. 2022 - 2023
Prof. Gabriele Tolomei

MSc in Computer Science

La Sapienza, University of Rome

### Author: Corsi Danilo (1742375) - corsi.1742375@studenti.uniroma1.it

# **Project details**
## **Introduction**
Bitcoin is a decentralized cryptocurrency, created in 2009 by an anonymous inventor under the pseudonym Satoshi Nakamoto. 
It does not have a central bank behind it that distributes new currency but relies on a network of nodes, i.e., PCs, that manage it in a distributed, peer-to-peer mode; and on the use of strong cryptography to validate and secure transactions. 
Transactions can be made through the Internet to anyone with a "bitcoin address" 
Bitcoin's value is determined by the market and the number of people using it. 
Its blockchain, or public ledger of transactions, is constantly updated and validated by nodes in the network.
The cryptocurrency Bitcoin has attracted the attention of many people in recent years. 
However, it's price fluctuation can be extremely unpredictable, which makes it difficult to predict when the right time to buy or sell this digital currency will be. 
In this context, prediction Bitcoin prices can be a competitive advantage for investors and traders, as it could allow them to make informed decisions on the right time to enter or exit the market.
In this project, I will analyze some machine learning techniques to understand, through the processing of historical data, how accurately the price of Bitcoin can be predicted and whether this can provide added value to cryptocurrency investors and traders.

## **Goal**
**``Is it possible to make predictions about the price of Bitcoin using machine learning methods in combination with the price information and technical characteristics of its blockchain?``**

## **Dataset**
I chose to collect data on the Bitcoin blockchain using the API of the website Blockchain.org and the price information from two famous exchange, Binance and Kraken. They were retrieved the most relevant information from the last four years to the present day (a period for which there were moments of high volatility but also a lot of price lateralization). The procedure has been made as automatic as possible so that the same periods are considered each time the entire procedure is executed. 

The features taken under consideration were divided into several categories:

- **Currency Statistics**
   - **ohlcv:** stands for “Open, High, Low, Close and Volume” and it's a list of the five types of data that are most common in financial analysis regarding price.
   - **market-price:** the average USD market price across major bitcoin exchanges.
   - **trade-volume-usd:** the total USD value of trading volume on major bitcoin exchanges.
   - **total-bitcoins:** the total number of mined bitcoin that are currently circulating on the network.
   - **market-cap:** the total USD value of bitcoin in circulation.

- **Block Details**
   - **blocks-size:** the total size of the blockchain minus database indexes in megabytes.
   - **avg-block-size:** the average block size over the past 24 hours in megabytes.
   - **n-transactions-total:** the total number of transactions on the blockchain.
   - **n-transactions-per-block:** the average number of transactions per block over the past 24 hours.

- **Mining Information**
   - **hash-rate:** the estimated number of terahashes per second the bitcoin network is performing in the last 24 hours.
   - **difficulty:** a relative measure of how difficult it is to mine a new block for the blockchain.
   - **miners-revenue:** total value of coinbase block rewards and transaction fees paid to miners.
   - **transaction-fees-usd:** the total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.

- **Network Activity**
   - **n-unique-addresses:** the total number of unique addresses used on the blockchain.
   - **n-transactions:** the total number of confirmed transactions per day.
   - **estimated-transaction-volume-usd:** the total estimated value in USD of transactions on the blockchain.


## **Project pipeline**

The project is structured like this:
- **Data crawling:** Bitcoin data retrieval via APIs call
- **Feature engineering:** manipulation, visualization and feature extraction
- **Models’ train / validation:** by performing hyperparameter tuning and cross validation based on different methods of splitting the dataset
- **Final scores:** testing the final models and compare the results

Project carried out with `Apache Spark` (but during feature engineering I converted the Spark dataframe to a Pandas one to make some plots)

## **Data crawling / Feature engineering**
After obtaining the features regarding the technical data of the blockchain and the price of Bitcoin by contacting the APIs of Blockchain.org and the two exchanges, other features are added:
*   **next-market-price:** represents the price of Bitcoin for the next day (this will be the target variable on which to make predictions).
*   **sma-x-days:** indicators that calculate the average price over a specified number of days (5, 7, 10, 20, 50 and 100 days in our case). They are commonly used by traders to identify trends and potential buy or sell signals.

All these features will be divided into two distinct groups:
- **Base features:** contains all the Currency Statistics features
- **Base and additional features:** contains the Base features plus the additional features divided based on their correlation value with the price: 
    - If >= 0.6, then then they will be considered **most correlated**.
    - If < 0.6, then then they will be considered **least correlated**.

The strategy for the model's train / validation phase will be:
- Test models with base features
- See if by adding the additional most and least correlated features to them improves the performance

The whole dataset will be splitted into two sets:
* **Train / Validation set:** will be used to train the models and validate the performances.
* **Test set:** will be used to perform price prediction on never-before-seen data (the last 3 months of the original dataset will be used).

## **Models train / validation**
During this phase the dataset will be splitted according to different splitting method (in order to figure out which one works best for our problem):

- **Block time series splits:** involves dividing the time series into blocks of equal length, and then using each block as a separate fold for cross-validation.

   ![block-splits.png](./notebooks/images/block-splits.png)

- **Walk forward time series splits:** involves using a sliding window approach to create the training and validation sets for each fold. The model is trained on a fixed window of historical data, and then validated on the next observation in the time series. This process is repeated for each subsequent observation, with the window sliding forward one step at a time. 

   ![walk-forward-splits.png](./notebooks/images/walk-forward-splits.png)

- **Single time series split** involves dividing the time series considering as validation set a narrow period of time and as train set everything that happened before this period, in such a way as to best benefit from the trend in the short term.

   ![single-split.png](./notebooks/images/single-split.png)

Several types of regression algorithms will be used, including: 
* **Linear Regression**
* **Generalized Linear Regression**
* **Random Forest Regressor**
* **Gradient Boosting Tree Regressor** 

To see their differences and how they perform in the various stages of training / validation and testing. 

Different types of metrics will be used, including: 
* **RMSE (Root Mean Squared Error)**
* **MSE (Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **MAPE (Mean Absolute Percentage Error)**
* **R2 (R-squared)**
* **Adjusted R2**

To get a complete picture of the performance of the various models.

Since predicting the price accurately is very difficult I will see also how good the models are at predicting whether the price will go up or down.
* I compute whether each prediction is correct (1) or not (0)
* Then I count the number of correct prediction
* Finally I compute the percentage of accuracy of the model

The train / validation pipeline will be:
- **Default without normalization:** make predictions using the base model.
- **Default with normalization:** like the previous one but features are normalized.

Then the features that gave on average the most satisfactory results (for each model) are chosen and proceeded with:
- **Hyperparameter tuning:** finding the best parameters to use. During this phase I compute a score for each parameter chosen by each split, assigning weights based on:
   * Their frequency for each split (if the same parameters are chosen from several splits, these will have greater weight) 
   * The split they belong to (the closer the split is to today's date the more weight they will have)
   * Their RMSE value for each split (the lower this is, the more weight they will have)
   
   Then, the overall score will be calculated by putting together these 3 weights for each parameter and the one with the best score will be the chosen parameter

- **Cross Validation:** validate the performance of the model with the chosen parameters.

If the final results are satisfactory, the model will be trained on the whole train / validation set and saved in order to make predictions on the test set.

## **Final scores**
After loading the previously trained models on the whole train / validation set, the test set is divided into further mini-sets of 1 week, 15 days, 1 month and 3 months to see how the models' performance degrades as time increases.

# **Project structure**

```
.
├── datasets
│  ├── output
│  │  ├── bitcoin_blockchain_data_15min_test.parquet
│  │  └── bitcoin_blockchain_data_15min_train_valid.parquet
│  ├── raw
│  │  └── bitcoin_blockchain_data_15min.parquet
│  └── temp
├── features
│  ├── all_features.json
│  ├── features_relevance.json
│  ├── least_rel_features.json
│  └── most_rel_features.json
├── models
│  ├── GeneralizedLinearRegression
│  ├── GradientBoostingTreeRegressor
│  ├── LinearRegression
│  └── RandomForestRegressor
├── notebooks
│  ├── 1 - Data crawling.ipynb
│  ├── 2 - Feature Engineering.ipynb
│  ├── 3.1 - [Block Split] Linear Regression.ipynb
│  ├── 3.2 - [Walk Forward Split]  Linear Regression.ipynb
│  ├── 3.3 - [Short Term Split] Linear Regression.ipynb
│  ├── 4.1 - [Block Split] Generalized Linear Regression.ipynb
│  ├── 4.2 - [Walk Forward Split] Generalized Linear Regression.ipynb
│  ├── 4.3 - [Short Term Split] Generalized Linear Regression.ipynb
│  ├── 5.1 - [Block Split] Random Forest Regressoripynb.ipynb
│  ├── 5.2 - [Walk Forward Split] Random Forest Regressoripynb.ipynb
│  ├── 5.3 - [Short Term Split] Random Forest Regressoripynb.ipynb
│  ├── 6.1 - [Block Split] Gradient Boosting Tree Regressor.ipynb
│  ├── 6.2 - [Walk Forward Split] Gradient Boosting Tree Regressor.ipynb
│  ├── 6.3 - [Short Term Split] Gradient Boosting Tree Regressor.ipynb
│  ├── 7 - Final scores.ipynb
│  └── images
│     ├── block-splits.png
│     ├── Drawings.excalidraw
│     ├── single-split.png
│     └── walk-forward-splits.png
├── presentation
│  ├── presentation.pptx
│  └── speech.docx
├── README.md
├── results
│  ├── block_splits
│  │  ├── GeneralizedLinearRegression.csv
│  │  ├── GeneralizedLinearRegression_accuracy.csv
│  │  ├── GradientBoostingTreeRegressor.csv
│  │  ├── GradientBoostingTreeRegressor_accuracy.csv
│  │  ├── LinearRegression.csv
│  │  ├── LinearRegression_accuracy.csv
│  │  ├── RandomForestRegressor.csv
│  │  └── RandomForestRegressor_accuracy.csv
│  ├── final
│  │  ├── final.csv
│  │  ├── final.png
│  │  ├── final_accuracy.csv
│  │  └── final_accuracy.png
│  ├── single_split
│  │  ├── GeneralizedLinearRegression.csv
│  │  ├── GeneralizedLinearRegression_accuracy.csv
│  │  ├── GradientBoostingTreeRegressor.csv
│  │  ├── GradientBoostingTreeRegressor_accuracy.csv
│  │  ├── LinearRegression.csv
│  │  ├── LinearRegression_accuracy.csv
│  │  ├── RandomForestRegressor.csv
│  │  └── RandomForestRegressor_accuracy.csv
│  └── walk_forward_splits
│     ├── GeneralizedLinearRegression.csv
│     ├── GeneralizedLinearRegression_accuracy.csv
│     ├── GradientBoostingTreeRegressor.csv
│     ├── GradientBoostingTreeRegressor_accuracy.csv
│     ├── LinearRegression.csv
│     ├── LinearRegression_accuracy.csv
│     ├── RandomForestRegressor.csv
│     └── RandomForestRegressor_accuracy.csv
└── utilities
   ├── imports.py
   ├── parameters.py
   └── utilities.py
```

### **Datasets folder: contains the original, temporary and processed datasets**
- **bitcoin_blockchain_data_15min_test.parquet:** dataset used in the final phase of the project to perform price prediction on never-before-seen data.
- **bitcoin_blockchain_data_15min_train_validation.parquet:** dataset used to train and validate the models.
- **bitcoin_blockchain_data_15min.parquet:** original dataset obtained by making calls to the APIs.

### **Features folder: contains the features used throughout the project**
- **base_and_least_corr_features.json:** contains the name of the currency features plus the least relevant features with respect to the price of Bitcoin.
- **base_and_most_corr_features.json:** contains the name of the currency features plus the most relevant features with respect to the price of Bitcoin.
- **base_features.json:** contains the name of the currency features of Bitcoin.

### **Models folder: contains files related to the trained models**
- Each folder **(GeneralizedLinearRegression, GradientBoostingTreeRegressor, LinearRegression and RandomForestRegressor)** contains the trained model with the best parameters, ready to be used to perform price prediction on never-before-seen data.

### **Notebooks folder: contains notebooks produced**
- **1-data-crawling.ipynb:** crawling data on Bitcoin's price and blochckain by querying APIs.
- **2-feature-engineering.ipynb:** adding useful features regardings the price of Bitcoin, visualizing data and performing feature selection.
- **3-5-splitting_method.ipynb:** it performs training/validation of models according to the chosen split method (block split, walk forward split or single split).
- **6-final-scores.ipynb:** display the final scores and making predictions on the test set with the models trained on the whole train / validation set.

### **Results folder: contains all results obtained**
- Based on the splitting method, results regarding metrics and accuracy are collected (including the final ones).

### **Utilities folder: contains files defined by me used by most notebooks to reuse the code**

- **imports.py:** contains imports of external libraries.
- **parameters.py:** contains the parameters used by the models during the train / validation phase.
- **utilities.py:** contains functions that are used by the models during the train / validation phase.
- **visualization.py:** contains methods for plotting graphs.