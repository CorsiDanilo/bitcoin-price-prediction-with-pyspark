# **Bitcoin price prediction with PySpark**
## Big Data Computing final project - A.Y. 2022 - 2023
Prof. Gabriele Tolomei

MSc in Computer Science

La Sapienza, University of Rome

### Author: Corsi Danilo (1742375) - corsi.1742375@studenti.uniroma1.it

# **Project details**
## **Outline**
- In this project I’ve decided to build a Bitcoin price forecasting model in order to see if it possible to make predictions about the price of Bitcoin using machine learning methods
- I will first introduce what bitcoin is and what is the aim of this project
- Next we will see what data will be used and how to achieve the goal
- Followed by a description of the main stages of the project
- And finally draw the final conclusions

## **Introduction**
- Bitcoin is a decentralized cryptocurrency, created in 2009 by an anonymous inventor under the pseudonym of Satoshi Nakamoto
- It does not have a central bank behind it but relies on a network of nodes that manage it in a distributed, peer-to-peer mode
- It uses strong cryptography to validate and secure transactions
- These can be made through the Internet to anyone with a bitcoin address
- And are contained in a public ledger of which is constantly updated and validated by nodes in the network
- It’s value is determined by the market and the number of people using it
- This criptocurrency has attracted the attention of many people in recent years, however, it's price fluctuation can be extremely unpredictable
- In this context, predicting Bitcoin prices can be a competitive advantage for investors and traders, as it could allow them to make informed decisions on the right time to enter or exit the market

## **Goal**
- ``Analyze some machine learning techniques to understand, through the processing of historical data, how accurately the price of Bitcoin can be predicted and whether this can provide added value to cryptocurrency investors and traders``

---

### ⚠️ **Note**: Because of the large size of the notebooks with the outputs containing the plots, it was not possible for me to upload them to the E-Learning / GitHub platforms, below are links to the notebooks with the outputs viewable using Colab
1.  [Data crawling](https://drive.google.com/file/d/1Kge0w40KDNjtRWsx0i82eDCg8ifTlAHX/view?usp=sharing)
2.  [Feature engineering](https://drive.google.com/file/d/1KvpnWJ5VqOhG0wBaG6Hn_49zI0pxxFt1/view?usp=sharing) 

3. **Block splitting:**

      3.1. [Linear Regression](https://drive.google.com/file/d/1L_TxiphKFR2qDamYs7fS8K3Ex2RONYR_/view?usp=sharing) 

      3.2. [Generalized Linear Regression](https://drive.google.com/file/d/1Lvd9TmYMYKbra3PFapNJY3-5Pa_LXIXq/view?usp=sharing) 

      3.3. [Random Forest Regressor](https://drive.google.com/file/d/1MCtofqa6kxOB1nAZH1QTSScmKZhmeaoA/view?usp=sharing) 

      3.4. [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1Lb8QZBAaTcL-kiZCPCyND7RiPuqoFlvb/view?usp=sharing) 

4.  **Walk forward splitting:**

      4.1 [Linear Regression](https://drive.google.com/file/d/1LxmudhmiPZ_FTfMT0rTT5Z8tSqdd-fDV/view?usp=sharing) 

      4.2 [Generalized Linear Regression](https://drive.google.com/file/d/1M5SglOmF780D8pBIwEfX-KjVRlD1RdZj/view?usp=sharing) 

      4.3 [Random Forest Regressor](https://drive.google.com/file/d/1LxF0qQHg-gZG9TQTI3MHH--oszB4BOaR/view?usp=sharing) 

      4.4 [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1M-wnVvVn2A5haQmoYINR3TmKaEKpTKm8/view?usp=sharing) 

5.  **Single splitting:** 

      5.1 [Linear Regression](https://drive.google.com/file/d/1MKPu0XQ66JIjeSFCcUeLzUDS9i0edQ3O/view?usp=sharing) 

      5.2 [Generalized Linear Regression](https://drive.google.com/file/d/1MLTxZ9VWbURL3tQaVA9p97njkClsf3rw/view?usp=sharing) 

      5.3 [Random Forest Regressor](https://drive.google.com/file/d/1MGlzaOP09SDFbdujooA0hxmgM7G5XByk/view?usp=sharing) 

      5.4 [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1MLSUwe35Ty5YP4zrhRwggq5jmk-hD894/view?usp=sharing) 

6. [Final scores](https://drive.google.com/file/d/1MEjsk7Dstbxe_qqjGwbES2JRupBtCnTF/view?usp=sharing) 

## **Dataset and features**
- I collected Bitcoin blockchain data using the API of the [Blockchain.com](https://www.blockchain.com/) website and price information from two popular exchanges, [Binance](https://www.binance.com) and [Kraken](https://www.kraken.com/)
- I decided to retreieve the most relevant data from the last four years to current days, a period for which there were moments of high volatility but also some price lateralization
- The features taken under consideration were divided into several categories, from those that describe the price characteristics to those that go into more detail about Bitcoin's blockchain:
   - **Currency Statistics**
      - `ohlcv:` stands for “Open, High, Low, Close and Volume” and it's a list of the five types of data that are most common in financial analysis regarding price.
      - `market-price:` the average USD market price across major bitcoin exchanges.
      - `trade-volume-usd:` the total USD value of trading volume on major bitcoin exchanges.
      - `total-bitcoins:` the total number of mined bitcoin that are currently circulating on the network.
      - `market-cap:` the total USD value of bitcoin in circulation.

   - **Block Details**
      - `blocks-size:` the total size of the blockchain minus database indexes in megabytes.
      - `avg-block-size:` the average block size over the past 24 hours in megabytes.
      - `n-transactions-total:` the total number of transactions on the blockchain.
      - `n-transactions-per-block:` the average number of transactions per block over the past 24 hours.

   - **Mining Information**
      - `hash-rate:` the estimated number of terahashes per second the bitcoin network is performing in the last 24 hours.
      - `difficulty:` a relative measure of how difficult it is to mine a new block for the blockchain.
      - `miners-revenue:` total value of coinbase block rewards and transaction fees paid to miners.
      - `transaction-fees-usd:` the total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.

   - **Network Activity**
      - `n-unique-addresses:` the total number of unique addresses used on the blockchain.
      - `n-transactions:` the total number of confirmed transactions per day.
      - `estimated-transaction-volume-usd:` the total estimated value in USD of transactions on the blockchain.

   <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/features_group.png?raw=1">

## **Project pipeline**
- The project is structured in this way
- First, I retrieved all the data and processed them in order to decide how to use the features
- Then different models are trained using different methods of splitting the dataset, which we will see later
- And then the final results are collected and conclusions are drawn
- The project was carried out with Apache Spark but during some phases I converted the Spark dataframe to a Pandas one to make some plots

### **1. Data crawling / Feature engineering**
**Features**
- After obtaining all the data, other features were added such as:
   - `next-market-price:` that represents the price of Bitcoin for the next day, on which predictions will be made
   - `simple-moving-averages:` indicators that calculate the average price over a specified number of days
- Then all the features have been divided into three distinct final groups:
   - `Base features:` contains all the price features
   - `Base + most / least correlated features:` contains the previous ones plus the additional blockchain features divided based on their correlation value with the price
   - If this value is greater than equal to 0.6 they will be considered most correlated, least correlated otherwise

   <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/grouped_features.png?raw=1">

**Splitting**
- Then the whole dataset will be splitted into two sets:
   - `Train / Validation set:` that will be used to train the models and validate the performances
   - `Test set:` that will be used to perform price prediction on never-before-seen data, in this case the last 3 months of the original dataset will be used

### **2. Models train / validation**
**Splitting methods**
- Three different splitting methods were used to train and validate the models in order to figure out which one works best for this problem
   - `Block splits` involves dividing the time series into blocks of equal length

      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/block-splits.png?raw=1">
   - `Walk forward splits` involves using a sliding window approach to create the training and validation sets

      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/walk-forward-splits.png?raw=1">
   - `Single split` involves dividing the time series considering a narrow period of time making a single split

      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/single-split.png?raw=1">
- In the latter case I consider only 2 years instead of 4 as in the others, so as to best benefit from the trend in the short term

**Models and metrics**
- Several types of regression algorithms between linear and tree-based will be tested to see their differences: 
   * `Linear Regression`
   * `Generalized Linear Regression`
   * `Random Forest Regressor`
   * `Gradient Boosting Tree Regressor` 

- Different types of metrics will be used to get a complete picture of the performance of the various models, including: 
   * `RMSE (Root Mean Squared Error)`
   * `MSE (Mean Squared Error)`
   * `MAE (Mean Absolute Error)`
   * `MAPE (Mean Absolute Percentage Error)`
   * `R2 (R-squared)`
   * `Adjusted R2`

**Accuracy**
- Since predicting the price accurately is very difficult, I tried to compute how good the models are at predicting whether the price will go up or down like this:
   - For each prediction, I am going to consider it correct if the actual price goes up or down and the predicted price follows that trend, wrong if vice versa
   - After that I count the number of correct predictions among all of them
   - And finally I compute the overall percentage of accuracy

   <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/accuracy_procedure.png?raw=1">

**Pipeline**
- Concern the train / validation pipeline, it is structured like this:
   - First of all, I saw how the `default models` behave with the three feature groups and applying normalisation to them or not

      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/base_model_procedure.png?raw=1">

   - Then the features that for each model gave the most satisfactory results are chosen and proceed with the `hyperparameter tuning` to find the best model’s parameters to use
   - Since during this stage will be used the Block split or  Walk forward split method of the dataset I compute a score for each set of parameters chosen by each split, assigning weights based on their `frequency of occurrence`, `split belonging` and `RMSE value`
   - Then, the overall score will be calculated by putting together these weights for each set of parameters and the one with the best score will be the chosen one
      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/hyper_param_tuning_procedure.png?raw=1">
   
   - After that, the performance of each model is validated by performing `cross validation`
   - And if the final results are satisfactory, the models will be trained on the whole train / validation set and saved in order to make predictions on the test set
      <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/cross_valid_and_final_procedure.png?raw=1">

### **3. Final scores**
- On this last phase, all results obtained up to that point are compared and final predictions on the test set are made
- This has been divided into further mini-sets of  to see how the models performance degrades as time increases

   <img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/notebooks/images/test_split_plot.png?raw=1">

---

⚠️ **Note**: Due to the large size of the notebooks with the outputs, it was not possible for me to upload them to the E-Learning / GitHub platforms, below are links to the notebooks with the outputs viewable using Colab: 

1.  [Data crawling](https://drive.google.com/file/d/1Kge0w40KDNjtRWsx0i82eDCg8ifTlAHX/view?usp=sharing)
2.  [Feature engineering](https://drive.google.com/file/d/1KvpnWJ5VqOhG0wBaG6Hn_49zI0pxxFt1/view?usp=sharing) 

3. **Block splitting:**

      3.1. [Linear Regression](https://drive.google.com/file/d/1L_TxiphKFR2qDamYs7fS8K3Ex2RONYR_/view?usp=sharing) 

      3.2. [Generalized Linear Regression](https://drive.google.com/file/d/1Lvd9TmYMYKbra3PFapNJY3-5Pa_LXIXq/view?usp=sharing) 

      3.3. [Random Forest Regressor](https://drive.google.com/file/d/1MCtofqa6kxOB1nAZH1QTSScmKZhmeaoA/view?usp=sharing) 

      3.4. [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1Lb8QZBAaTcL-kiZCPCyND7RiPuqoFlvb/view?usp=sharing) 

4.  **Walk forward splitting:**

      4.1 [Linear Regression](https://drive.google.com/file/d/1LxmudhmiPZ_FTfMT0rTT5Z8tSqdd-fDV/view?usp=sharing) 

      4.2 [Generalized Linear Regression](https://drive.google.com/file/d/1M5SglOmF780D8pBIwEfX-KjVRlD1RdZj/view?usp=sharing) 

      4.3 [Random Forest Regressor](https://drive.google.com/file/d/1LxF0qQHg-gZG9TQTI3MHH--oszB4BOaR/view?usp=sharing) 

      4.4 [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1M-wnVvVn2A5haQmoYINR3TmKaEKpTKm8/view?usp=sharing) 

5.  **Single splitting:** 

      5.1 [Linear Regression](https://drive.google.com/file/d/1MKPu0XQ66JIjeSFCcUeLzUDS9i0edQ3O/view?usp=sharing) 

      5.2 [Generalized Linear Regression](https://drive.google.com/file/d/1MLTxZ9VWbURL3tQaVA9p97njkClsf3rw/view?usp=sharing) 

      5.3 [Random Forest Regressor](https://drive.google.com/file/d/1MGlzaOP09SDFbdujooA0hxmgM7G5XByk/view?usp=sharing) 

      5.4 [Gradient Boosting Tree Regressor](https://drive.google.com/file/d/1MLSUwe35Ty5YP4zrhRwggq5jmk-hD894/view?usp=sharing) 

6. [Final scores](https://drive.google.com/file/d/1MEjsk7Dstbxe_qqjGwbES2JRupBtCnTF/view?usp=sharing) 

# **Project structure**
```
.
|-- README.md
|-- datasets
|   |-- output
|   |   |-- bitcoin_blockchain_data_15min_test.parquet
|   |   `-- bitcoin_blockchain_data_15min_train_valid.parquet
|   |-- raw
|   |   `-- bitcoin_blockchain_data_15min.parquet
|   `-- temp
|-- features
|   |-- base_and_least_corr_features.json
|   |-- base_and_most_corr_features.json
|   `-- base_features.json
|-- models
|   |-- GeneralizedLinearRegression
|   |-- GradientBoostingTreeRegressor
|   |-- LinearRegression
|   `-- RandomForestRegressor
|-- notebooks
|   |-- 1-data-crawling.ipynb
|   |-- 2-feature-engineering.ipynb
|   |-- 3-block-split_GeneralizedLinearRegression.ipynb
|   |-- 3-block-split_GradientBoostingTreeRegressor.ipynb
|   |-- 3-block-split_LinearRegression.ipynb
|   |-- 3-block-split_RandomForestRegressor.ipynb
|   |-- 4-walk-forward-split_GeneralizedLinearRegression.ipynb
|   |-- 4-walk-forward-split_GradientBoostingTreeRegressor.ipynb
|   |-- 4-walk-forward-split_LinearRegression.ipynb
|   |-- 4-walk-forward-split_RandomForestRegressor.ipynb
|   |-- 5-single-split_GeneralizedLinearRegression.ipynb
|   |-- 5-single-split_GradientBoostingTreeRegressor.ipynb
|   |-- 5-single-split_LinearRegression.ipynb
|   |-- 5-single-split_RandomForestRegressor.ipynb
|   |-- 6-final-scores.ipynb
|   `-- images
|       |-- Drawings.excalidraw
|       |-- block-splits.png
|       |-- single-split.png
|       `-- walk-forward-splits.png
|-- presentation
|   |-- presentation.pptx
|-- requirements.txt
|-- results
|   |-- block_splits
|   |   |-- GeneralizedLinearRegression_accuracy.csv
|   |   |-- GeneralizedLinearRegression_all.csv
|   |   |-- GeneralizedLinearRegression_rel.csv
|   |   |-- GradientBoostingTreeRegressor_accuracy.csv
|   |   |-- GradientBoostingTreeRegressor_all.csv
|   |   |-- GradientBoostingTreeRegressor_rel.csv
|   |   |-- LinearRegression_accuracy.csv
|   |   |-- LinearRegression_all.csv
|   |   |-- LinearRegression_rel.csv
|   |   |-- RandomForestRegressor_accuracy.csv
|   |   |-- RandomForestRegressor_all.csv
|   |   `-- RandomForestRegressor_rel.csv
|   |-- final
|   |   |-- final.csv
|   |   `-- plots
|   |       |-- default_train_val_r2.png
|   |       |-- default_train_val_r2_non_negative.png
|   |       |-- default_train_val_rmse.png
|   |       |-- final_test_accuracy.png
|   |       |-- final_test_fifteen_days_prediction.png
|   |       |-- final_test_one_month_prediction.png
|   |       |-- final_test_one_week_prediction.png
|   |       |-- final_test_r2.png
|   |       |-- final_test_r2_non_negative.png
|   |       |-- final_test_rmse.png
|   |       |-- final_test_three_months_prediction.png
|   |       |-- final_train_val_accuracy.png
|   |       |-- final_train_val_r2.png
|   |       |-- final_train_val_r2_non_negative.png
|   |       `-- final_train_val_rmse.png
|   |-- single_split
|   |   |-- GeneralizedLinearRegression_accuracy.csv
|   |   |-- GeneralizedLinearRegression_all.csv
|   |   |-- GeneralizedLinearRegression_rel.csv
|   |   |-- GradientBoostingTreeRegressor_accuracy.csv
|   |   |-- GradientBoostingTreeRegressor_all.csv
|   |   |-- GradientBoostingTreeRegressor_rel.csv
|   |   |-- LinearRegression_accuracy.csv
|   |   |-- LinearRegression_all.csv
|   |   |-- LinearRegression_rel.csv
|   |   |-- RandomForestRegressor_accuracy.csv
|   |   |-- RandomForestRegressor_all.csv
|   |   `-- RandomForestRegressor_rel.csv
|   `-- walk_forward_splits
|       |-- GeneralizedLinearRegression_accuracy.csv
|       |-- GeneralizedLinearRegression_all.csv
|       |-- GeneralizedLinearRegression_rel.csv
|       |-- GradientBoostingTreeRegressor_accuracy.csv
|       |-- GradientBoostingTreeRegressor_all.csv
|       |-- GradientBoostingTreeRegressor_rel.csv
|       |-- LinearRegression_accuracy.csv
|       |-- LinearRegression_all.csv
|       |-- LinearRegression_rel.csv
|       |-- RandomForestRegressor_accuracy.csv
|       |-- RandomForestRegressor_all.csv
|       `-- RandomForestRegressor_rel.csv
`-- utilities
    |-- config.py
    |-- feature_engineering_utilities.py
    |-- final_scores_utilities.py
    |-- imports.py
    |-- train_validation_utilities.py
```
### `Datasets folder:` contains the original, temporary and processed datasets
- `bitcoin_blockchain_data_15min_test.parquet:` dataset used in the final phase of the project to perform price prediction on never-before-seen data
- `bitcoin_blockchain_data_15min_train_validation.parquet:` dataset used to train and validate the models
- `bitcoin_blockchain_data_15min.parquet:` original dataset obtained by making calls to the APIs

### `Features folder:` contains the features used throughout the project
- `base_and_least_corr_features.json:` contains the name of the currency features plus the least relevant features with respect to the price of Bitcoin
- `base_and_most_corr_features.json:` contains the name of the currency features plus the most relevant features with respect to the price of Bitcoin
- `base_features.json:` contains the name of the currency features of Bitcoin

### `Models folder:` contains files related to the trained models
- Each folder (`GeneralizedLinearRegression`, `GradientBoostingTreeRegressor`, `LinearRegression` and `RandomForestRegressor`) contains the trained model with the best parameters, ready to be used to perform price prediction on never-before-seen data

### `Notebooks folder:` contains notebooks produced
- `1-data-crawling.ipynb:` crawling data on Bitcoin's price and blochckain by querying APIs
- `2-feature-engineering.ipynb:` adding useful features regardings the price of Bitcoin, visualizing data and performing feature selection
- `3-5-<splitting-method>_<model>.ipynb:` it performs training/validation of models according to the chosen split method (block split, walk forward split or single split)
- `6-final-scores.ipynb:` display the final scores and making predictions on the test set with the models trained on the whole train / validation set

### `Results folder:` contains all results obtained
- Based on the splitting method, results regarding metrics and accuracy are collected (including the final ones).

### `Utilities folder:` contains files defined by me used by most notebooks to reuse the code
- `config.py` contains global variables that can be used throughout the project
- `feature_engineering_utilities.py:` contains the methods used in the feature engineering notebook
- `final_scores_utilities.py:` contains the methods used in the notebook of final scores
- `imports.py:` contains imports of external libraries
- `train_validation_utilities.py:` contains the methods used in the notebooks where models are trained and validated

# **Final results**
<img src="https://github.com/CorsiDanilo/bitcoin-price-prediction-with-pyspark/blob/main/results/final/plots/final_test_predictions.jpg?raw=1">
