import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline

#Install some useful dependencies
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

import plotly.express as px

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import gc

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

# Define the evaluation metrics
# Notice that r2_adj metric is included when calculating r2
metrics = ['mse', 'rmse', 'mae', 'r2']

# --------------------------------------------------------------------------- # 

'''
Description: Split and keep the original time-series order
Args:
    dataSet: The dataSet which needs to be splited
    proportion: A number represents the split proportion

Return: 
    train_data: The train dataSet
    test_data: The test dataSet
'''
def trainSplit(dataset, proportion):
    records_num = dataset.count()
    split_point = int(records_num * proportion)
    
    train_data = dataset.filter(dataset.index < split_point)
    test_data = dataset.filter(dataset.index >= split_point)
    
    return (train_data,test_data)

import time
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

def compute_results(predictions, label_col, ml_model, proportion, param, start, end):
    # Compute test error by several evaluators
    rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='rmse')
    mae_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='mae')
    r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='r2')
    var_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='var')
    
    predictions_pd = predictions.select(label_col,"prediction").toPandas()
    mape = mean_absolute_percentage_error(predictions_pd[label_col], predictions_pd["prediction"])
    
    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    var = var_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)
    # Adjusted R-squared
    n = predictions.count()
    p = len(predictions.columns)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    results = {
                "Model": ml_model,
                "Proportion": proportion,
                "Parameters": [list(param.values())],
                "RMSE": rmse,
                "MAPE":mape,
                "MAE": mae,
                "Variance": var,
                "R2": r2,
                "Adjusted_R2": adj_r2,
                "Time": end - start,
                "Predictions": predictions.select(label_col,"prediction",'timestamp')
            }

    return results

'''
Description: Use Grid Search to tune the Model 
Args:
    dataSet: The dataSet which needs to be splited
    proportion_lst: A list represents the split proportion
    feature_col: The column name of features
    label_col: The column name of label
    ml_model: The module to use
    params: Parameters which want to test 
    assembler: An assembler to dataSet
Return: 
    results_df: The best result in a pandas dataframe
'''
def autoTuning(dataSet, proportion_lst, feature_col, label_col, ml_model, params, assembler):
    # Initialize the best result for comparison
    train_result_lst = []    
    test_result_lst = []

    train_best_result = {"RMSE": float('inf')}
    test_best_result = {"RMSE": float('inf')}
    train_best_df = {}
    test_best_df = {}
    train_best_predictions_df = {}
    test_best_predictions_df = {}
    
    # Try different proportions 
    for proportion in proportion_lst:
        # Split the dataSet
        train_data,test_data = trainSplit(dataSet, proportion)
    
        # Cache it
        train_data.cache()
        test_data.cache()
    
        # ALL combination of params
        param_lst = [dict(zip(params, param)) for param in product(*params.values())]

        for param in param_lst:
            # Chosen Model
            if ml_model == "LinearRegression":
                model = LinearRegression(featuresCol=feature_col, \
                                         labelCol=label_col, \
                                         maxIter=param['maxIter'], \
                                         regParam=param['regParam'], \
                                         elasticNetParam=param['elasticNetParam'])
            
            elif ml_model == "RandomForest":
                model = RandomForestRegressor(featuresCol=feature_col, \
                                              labelCol=label_col, \
                                              numTrees = param["numTrees"], \
                                              maxDepth = param["maxDepth"])
            
            elif ml_model == "GBTRegression":
                model = GBTRegressor(featuresCol=feature_col, \
                                     labelCol=label_col, \
                                     maxIter = param['maxIter'], \
                                     maxDepth = param['maxDepth'], \
                                     stepSize = param['stepSize'])

            # Chain assembler and model in a Pipeline
            pipeline = Pipeline(stages=[assembler, model])
            # Train a model and calculate running time
            start = time.time()
            pipeline_model = pipeline.fit(train_data)
            end = time.time()

            # Make predictions
            train_predictions = pipeline_model.transform(train_data)
            test_predictions = pipeline_model.transform(test_data)

            train_results = compute_results(train_predictions, label_col, ml_model, proportion, param, start, end)
            test_results = compute_results(test_predictions, label_col, ml_model, proportion, param, start, end)

            # Store each splits result
            train_result_lst.append(train_results)
            test_result_lst.append(test_results)
                
            # Only store the lowest RMSE
            if test_results['RMSE'] < test_best_result['RMSE']:
                train_best_result = train_results
                test_best_result = test_results
                train_best_df = train_data
                test_best_df = test_data
                train_best_predictions_df = train_predictions
                test_best_predictions_df = test_predictions
                
        # Release Cache
        train_data.unpersist()
        test_data.unpersist()
        
    # Transform dict to pandas dataframe
    train_result_pd = pd.DataFrame(train_result_lst)
    test_result_pd = pd.DataFrame(test_result_lst)

    train_best_result_df = pd.DataFrame(train_best_result)
    test_best_result_df = pd.DataFrame(test_best_result)
    train_best_df = train_data.toPandas()
    test_best_df = test_data.toPandas()
    train_best_predictions_df = train_predictions.toPandas()
    test_best_predictions_df = test_predictions.toPandas()

    # return train_best_result_df, test_best_result_df, train_best_df, test_best_df, train_best_predictions_df, test_best_predictions_df
    return train_result_pd, test_result_pd, train_best_df, test_best_df, train_best_predictions_df, test_best_predictions_df

# --------------------------------------------------------------- #

'''
Description: Multiple Splits Cross Validation on Time Series data
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def mulTsCrossValidation(num, n_splits):
    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(1, n_splits+1):
        # Calculate train size and test size
        train_size = i * num // (n_splits + 1) + num % (n_splits + 1)
        test_size = num //(n_splits + 1)

        # Calculate the start/split/end point for each fold
        start = 0
        split = train_size
        end = train_size + test_size
        
        # Avoid to beyond the whole number of dataSet
        if end > num:
            end = num
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Blocked Time Series Cross Validation
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def blockedTsCrossValidation(num, n_splits):
    kfold_size = num // n_splits

    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(n_splits):
        # Calculate the start/split/end point for each fold
        start = i * kfold_size
        end = start + kfold_size
        # Manually set train-test split proportion in each fold
        split = int(0.8 * (end - start)) + start
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Walk Forward Validation on Time Series data
Args:
    num: Number of DataSet
    min_obser: Minimum Number of Observations
    expand_window: Sliding or Expanding Window
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def wfTsCrossValidation(num, min_obser, expand_window):
    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(min_obser,num,expand_window):
        # Calculate the start/split/end point for each fold
        start = 0
        split = i
        end = split + expand_window
        
        # Avoid to beyond the whole number of dataSet
        if end > num:
            end = num
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Cross Validation on Time Series data
Args:
    dataSet: The dataSet which needs to be splited
    feature_col: The column name of features
    label_col: The column name of label
    ml_model: The module to use
    params: Parameters which want to test 
    assembler: An assembler to dataSet
    cv_info: The type of Cross Validation
Return: 
    tsCv_df: All the splits performance of each model in a pandas dataframe
'''
def tsCrossValidation(dataSet, feature_col, label_col, ml_model, params, assembler, cv_info):
    # Get the number of samples
    num = dataSet.count()
    
    # Save results in a list
    train_result_lst = []    
    test_result_lst = []

    # Initialize the best result for comparison
    result_best = {"RMSE": float('inf')}
    train_best = pd.DataFrame()
    test_best = pd.DataFrame()
    predictions_best = pd.DataFrame()

    # ALL combination of params
    param_lst = [dict(zip(params, param)) for param in product(*params.values())]

    for param in param_lst:
        # Chosen Model
        if ml_model == "LinearRegression":
            model = LinearRegression(featuresCol=feature_col, \
                                     labelCol=label_col, \
                                     maxIter=param['maxIter'], \
                                     regParam=param['regParam'], \
                                     elasticNetParam=param['elasticNetParam'])

        elif ml_model == "RandomForest":
            model = RandomForestRegressor(featuresCol=feature_col, \
                                          labelCol=label_col, \
                                          numTrees = param["numTrees"], \
                                          maxDepth = param["maxDepth"])

        elif ml_model == "GBTRegression":
            model = GBTRegressor(featuresCol=feature_col, \
                                 labelCol=label_col, \
                                 maxIter = param['maxIter'], \
                                 maxDepth = param['maxDepth'], \
                                 stepSize = param['stepSize'])
        
        if cv_info['cv_type'] == 'blkTs':
             split_position_df = blockedTsCrossValidation(num, cv_info['kSplits'])

        for position in split_position_df.itertuples():
            # Get the start/split/end position from a kind of Time Series Cross Validation
            start = getattr(position, 'start')
            splits = getattr(position, 'split')
            end = getattr(position, 'end')
            idx  = getattr(position, 'Index')
            
            # Train/Test size
            train_size = splits - start
            test_size = end - splits

            # Get training data and test data
            train_data = dataSet.filter(dataSet.index.between(start, splits-1))
            test_data = dataSet.filter(dataSet.index.between(splits, end-1))

            # Cache it
            train_data.cache()
            test_data.cache()

            # Chain assembler and model in a Pipeline
            pipeline = Pipeline(stages=[assembler, model])
            # Train a model and calculate running time
            start = time.time()
            pipeline_model = pipeline.fit(train_data)
            end = time.time()

            # Make predictions
            train_predictions = pipeline_model.transform(train_data)
            test_predictions = pipeline_model.transform(test_data)

            train_results = compute_results_cv(train_predictions, label_col, ml_model, idx, param, cv_info, start, end)
            test_results = compute_results_cv(test_predictions, label_col, ml_model, idx, param, start, end)

            # Store each splits result
            train_result_lst.append(train_results)
            test_result_lst.append(test_results)



            # Compute test error by several evaluator
            rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='rmse')
            mae_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='mae')
            r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='r2')
            var_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='var')
            
            predictions_pd = predictions.select("timestamp",label_col,"prediction").toPandas()
            mape = mean_absolute_percentage_error(predictions_pd[label_col], predictions_pd["prediction"])

            rmse = rmse_evaluator.evaluate(predictions)
            mae = mae_evaluator.evaluate(predictions)
            var = var_evaluator.evaluate(predictions)
            r2 = r2_evaluator.evaluate(predictions)
            # Adjusted R-squared
            n = predictions.count()
            p = len(predictions.columns)
            adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

            # Use dict to store each result
            results = {
                "Model": ml_model,
                'CV_type': cv_info['cv_type'],
                "Splits": idx + 1,
                "Train&Test": (train_size,test_size),
                "Parameters": list(param.values()),
                "RMSE": rmse,
                "MAPE": mape,
                "MAE": mae,
                "Variance": var,
                "R2": r2,
                "Adjusted_R2": adj_r2,
                "Time": end - start
            }
            
            # Store each splits result
            result_lst.append(results)

            # Only store the lowest RMSE
            if results['RMSE'] < result_best['RMSE']:
                result_best = results
                train_best = train_data.toPandas()
                test_best = test_data.toPandas()
                predictions_best = predictions.toPandas()
            
            # Release Cache
            train_data.unpersist()
            test_data.unpersist()

    # Transform dict to pandas dataframe
    tsCv_df = pd.DataFrame(result_lst)
    results_df = pd.DataFrame(result_best)
    return tsCv_df, result_best, train_best, test_best, predictions_best

# --------------------------------------------------------------- #

'''
Description: Displays the dataframes in the form of a timeseries graph
Args:
    train: Dataset related to training set
    valid: Dataset related to validation set
    training: Predictions result on the training set
    pred: Predictions result on the validation set
'''
def show_results(train, valid, train_pred, valid_pred):
  trace1 = go.Scatter(
      x = train['timestamp'],
      y = train['market-price'].astype(float),
      mode = 'lines',
      name = 'Train set'
  )

  trace2 = go.Scatter(
      x = valid['timestamp'],
      y = valid['market-price'].astype(float),
      mode = 'lines',
      name = 'Valid set'
  )

  trace3 = go.Scatter(
      x = train_pred['timestamp'],
      y = train_pred['prediction'].astype(float),
      mode = 'lines',
      name = 'Train predictions'
  )

  trace4 = go.Scatter(
      x = valid_pred['timestamp'],
      y = valid_pred['prediction'].astype(float),
      mode = 'lines',
      name = 'Valid prediction'
  )

  layout = dict(
      title='Train, valid and their predictions with Rangeslider',
      xaxis=dict(
          rangeselector=dict(
              buttons=list([
                  #change the count to desired amount of months.
                  dict(count=1,
                      label='1m',
                      step='month',
                      stepmode='backward'),
                  dict(count=6,
                      label='6m',
                      step='month',
                      stepmode='backward'),
                  dict(count=12,
                      label='1y',
                      step='month',
                      stepmode='backward'),
                  dict(count=36,
                      label='3y',
                      step='month',
                      stepmode='backward'),
                  dict(step='all')
              ])
          ),
          rangeslider=dict(
              visible = True
          ),
          type='date'
      )
  )

  data = [trace1,trace2,trace3,trace4]
  fig = dict(data=data, layout=layout)
  iplot(fig, filename = "Train, valid and their predictions with Rangeslider")