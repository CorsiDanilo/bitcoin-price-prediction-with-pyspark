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

# Apache Spark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StandardScaler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Python
import numpy as np
import pandas as pd
from itertools import product
import time

# Graph packages
# https://plotly.com/python/getting-started/#jupyterlab-support
# https://plotly.com/python/time-series/
import plotly.express as px

# Scikit-learn
from sklearn.metrics import mean_absolute_percentage_error

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

##########
# GLOBAL #
##########
metrics = ['mse', 'rmse', 'mae', 'r2']

##########
# SHARED #
##########

# Return the dataset with the selected features
def select_features(dataset, features, featureCol, labelCol):
  vectorAssembler = VectorAssembler(
    inputCols = features,
    outputCol = featureCol)

  dataset = vectorAssembler.transform(dataset)
  dataset = dataset.select(['timestamp','id', featureCol, labelCol])

  return dataset

'''
Description: Split and keep the original time-series order
Args:
    dataSet: The dataSet which needs to be splited
    proportion: A number represents the split proportion

Return: 
    train_data: The train dataSet
    valid_data: The validation dataSet
'''
def trainSplit(dataSet, proportion):
    records_num = dataSet.count()
    split_point = int(records_num * proportion)
    
    train_data = dataSet.filter(dataSet['id'] < split_point)
    valid_data = dataSet.filter(dataSet['id'] >= split_point)

    return (train_data,valid_data)

def show_results(train, valid, pred):
  trace1 = go.Scatter(
      x = train['timestamp'],
      y = train['tomorrow-market-price'].astype(float),
      mode = 'lines',
      name = 'Train set'
  )

  trace2 = go.Scatter(
      x = valid['timestamp'],
      y = valid['tomorrow-market-price'].astype(float),
      mode = 'lines',
      name = 'Valid set'
  )

  trace3 = go.Scatter(
      x = pred['timestamp'],
      y = pred['prediction'].astype(float),
      mode = 'lines',
      name = 'Valid prediction'
  )

  layout = dict(
      title='Train, valid and predictions with Rangeslider',
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

  data = [trace1,trace2,trace3]
  fig = dict(data=data, layout=layout)
  iplot(fig, filename = "Train, valid and predictions with Rangeslider")

'''
Description: Apply calculations on Time Series Cross Validation results to form the final Model Comparison Table
Args:
    cv_result: The results from tsCrossValidation()
    model_info: The model information which you would like to show
    evaluator_lst: The evaluator metrics which you would like to show
Return:
    comparison_df: A pandas dataframe of a model on a type of Time Series Cross Validation
'''
def modelComparison(cv_result, model_info, evaluator_lst):
    # Calculate mean of all splits on chosen evaluator
    col_mean_df = cv_result[evaluator_lst].mean().to_frame().T
    # Extract model info
    model_info_df = cv_result[model_info][:1]
    # Concatenate by row
    comparison_df = pd.concat([model_info_df,col_mean_df],axis=1)
    return comparison_df

#################
# SIMPLE MODELS #
#################

def simple_model(dataframe, modelName, featureCol, labelCol):
    if (modelName == 'LinearRegression'):
        model = LinearRegression(featuresCol=featureCol, labelCol=labelCol)
    if (modelName == 'GeneralizedLinearRegression'):
        model = GeneralizedLinearRegression(featuresCol=featureCol, labelCol=labelCol)
    elif (modelName == 'RandomForestRegressor'):
        model = RandomForestRegressor(featuresCol=featureCol, labelCol=labelCol)
    elif (modelName == 'GBTRegressor'):
        model = GBTRegressor(featuresCol=featureCol, labelCol=labelCol)

    fit_model = model.fit(dataframe)

    return fit_model

# Function to compute the r2 adjusted metric
# r2 is the r2 metric, n is the number of observations, k is the number of features
def compute_r2adj(r2, n, k):
  return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# Function to evaluate a model
def evaluate_model(predictions, modelName, typeName, label, prediction, metrics):
  r2 = None
  for metric in metrics:
    evaluator = RegressionEvaluator(labelCol=label, predictionCol=prediction, metricName=metric)
    evaluation = evaluator.evaluate(predictions)
    print(metric.upper()+' for '+modelName+' on '+typeName+' set: '+str(evaluation))
    if metric == 'r2':
      print('R2_adj'+' for '+modelName+' on '+typeName+' set: '+str(compute_r2adj(evaluation, predictions.count(), len(predictions.columns))))

# Function that create simple models (without hyperparameter tuning) and evaluate them
def evaluate_simple_model(dataframe, features, modelName, featureCol, labelCol, metrics = ['rmse', 'r2']):    
    # Select train and valid data features
    dataframe = select_features(dataframe, features, featureCol, labelCol)

    train_data, valid_data = trainSplit(dataframe, 0.8)

    # Train the model
    model = simple_model(train_data, modelName, featureCol, labelCol)
    
    # Model evaluation
    predictions = model.transform(valid_data)
    evaluate_model(predictions, modelName, 'validation', labelCol, 'prediction', metrics)

    # Show results
    show_results(train_data.toPandas(), valid_data.toPandas(), predictions.toPandas())

#########################
# HYPERPARAMETER TUNING #
#########################
def autoTuning(dataSet, features, params, proportion_lst, ml_model, feature_col, label_col):    
    dataSet = select_features(dataSet, features, feature_col, label_col)

    # Initialize the best result for comparison
    result_best = {"RMSE": float('inf')}
        
    # Try different proportions 
    for proportion in proportion_lst:
        # Split the dataSet
        train_data,valid_data = trainSplit(dataSet, proportion)
    
        # Cache it
        train_data.cache()
        valid_data.cache()
    
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
                
            elif ml_model == "GeneralizedLinearRegression":
                model = GeneralizedLinearRegression(featuresCol=feature_col, \
                                                    labelCol=label_col, \
                                                    maxIter=param['maxIter'], \
                                                    regParam=param['regParam'], \
                                                    family=param['family'], \
                                                    link=param['link'])

            elif ml_model == "RandomForestRegressor":
                model = RandomForestRegressor(featuresCol=feature_col, \
                                              labelCol=label_col, \
                                              numTrees = param["numTrees"], \
                                              maxDepth = param["maxDepth"])

            elif ml_model == "GBTRegressor":
                model = GBTRegressor(featuresCol=feature_col, \
                                     labelCol=label_col, \
                                     maxIter = param['maxIter'], \
                                     maxDepth = param['maxDepth'], \
                                     stepSize = param['stepSize'], \
                                     seed=0)
            
            # Chain assembler and model in a Pipeline
            pipeline = Pipeline(stages=[model])
            # Train a model and calculate running time
            start = time.time()
            pipeline_model = pipeline.fit(train_data)
            end = time.time()

            # Make predictions
            predictions = pipeline_model.transform(valid_data)

            # Compute validation error by several evaluators
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
        
            # Use dict to store each result
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

            # DEBUG: show data for each split
            # show_results(train_data.toPandas(), valid_data.toPandas(), predictions.toPandas())
            
            # Only store the lowest RMSE
            if results['RMSE'] < result_best['RMSE']:
                result_best = results

        # Release Cache
        train_data.unpersist()
        valid_data.unpersist()
        
    # Transform dict to pandas dataframe
    results_df = pd.DataFrame(result_best)

    return results_df

####################
# CROSS VALIDATION #
####################

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
        # Calculate train size and validation size
        train_size = i * num // (n_splits + 1) + num % (n_splits + 1)
        valid_size = num //(n_splits + 1)

        # Calculate the start/split/end point for each fold
        start = 0
        split = train_size
        end = train_size + valid_size
        
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
        # Manually set train-validation split proportion in each fold
        split = int(0.8 * (end - start)) + start
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
def tsCrossValidation(dataSet, features, params, cv_info, ml_model, feature_col, label_col):
    dataSet = select_features(dataSet, features, feature_col, label_col)

    # Get the number of samples
    num = dataSet.count()
    
    # Save results in a list
    result_lst = []
    trained_models = []

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
            
        elif ml_model == "GeneralizedLinearRegression":
                model = GeneralizedLinearRegression(featuresCol=feature_col, \
                                                    labelCol=label_col, \
                                                    maxIter=param['maxIter'], \
                                                    regParam=param['regParam'], \
                                                    family=param['family'], \
                                                    link=param['link'])

        elif ml_model == "RandomForestRegressor":
            model = RandomForestRegressor(featuresCol=feature_col, \
                                          labelCol=label_col, \
                                          numTrees = param["numTrees"], \
                                          maxDepth = param["maxDepth"])

        elif ml_model == "GBTRegressor":
            model = GBTRegressor(featuresCol=feature_col, \
                                 labelCol=label_col, \
                                 maxIter = param['maxIter'], \
                                 maxDepth = param['maxDepth'], \
                                 stepSize = param['stepSize'])
        
        # Identify the type of Cross Validation 
        if cv_info['cv_type'] == 'mulTs':
            split_position_df = mulTsCrossValidation(num, cv_info['kSplits'])
        elif cv_info['cv_type'] == 'blkTs':
            split_position_df = blockedTsCrossValidation(num, cv_info['kSplits'])

        for position in split_position_df.itertuples():
            # Get the start/split/end position from a kind of Time Series Cross Validation
            start = getattr(position, 'start')
            splits = getattr(position, 'split')
            end = getattr(position, 'end')
            idx  = getattr(position, 'Index')
            
            # Train/Validation size
            train_size = splits - start
            valid_size = end - splits

            # Get training data and validation data
            train_data = dataSet.filter(dataSet['id'].between(start, splits-1))
            valid_data = dataSet.filter(dataSet['id'].between(splits, end-1))

            # Cache it
            train_data.cache()
            valid_data.cache()

            # Chain assembler and model in a Pipeline
            pipeline = Pipeline(stages=[model])
            # Train a model and calculate running time
            start = time.time()
            pipeline_model = pipeline.fit(train_data)
            end = time.time()

            # Make predictions
            predictions = pipeline_model.transform(valid_data)

            # Compute validation error by several evaluator
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

            # Use dict to store each result
            results = {
                "Model": ml_model,
                'CV_type': cv_info['cv_type'],
                "Splits": idx + 1,
                "Train&Validation": (train_size,valid_size),
                "Parameters": list(param.values()),
                "RMSE": rmse,
                "MAPE": mape,
                "MAE": mae,
                "Variance": var,
                "R2": r2,
                "Adjusted_R2": adj_r2,
                "Time": end - start
            }

            # DEBUG: show data for each split
            # show_results(train_data.toPandas(), valid_data.toPandas(), predictions.toPandas())
            
            # Store each splits result
            result_lst.append(results)

            # Append the trained model to the list
            trained_models.append(pipeline_model)
            
            # Release Cache
            train_data.unpersist()
            valid_data.unpersist()

    # Transform dict to pandas dataframe
    tsCv_df = pd.DataFrame(result_lst)
    return tsCv_df, trained_models

#####################
# TRAIN FINAL MODEL #
#####################

def train_final_model(dataSet, features, params, ml_model, feature_col, label_col):    
    dataSet = select_features(dataSet, features, feature_col, label_col)

    # Split the dataSet
    train_data,valid_data = trainSplit(dataSet, 1)

    # Cache it
    train_data.cache()
    valid_data.cache()
    
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
            
        elif ml_model == "GeneralizedLinearRegression":
            model = GeneralizedLinearRegression(featuresCol=feature_col, \
                                                labelCol=label_col, \
                                                maxIter=param['maxIter'], \
                                                regParam=param['regParam'], \
                                                family=param['family'], \
                                                link=param['link'])

        elif ml_model == "RandomForestRegressor":
            model = RandomForestRegressor(featuresCol=feature_col, \
                                            labelCol=label_col, \
                                            numTrees = param["numTrees"], \
                                            maxDepth = param["maxDepth"])

        elif ml_model == "GBTRegressor":
            model = GBTRegressor(featuresCol=feature_col, \
                                    labelCol=label_col, \
                                    maxIter = param['maxIter'], \
                                    maxDepth = param['maxDepth'], \
                                    stepSize = param['stepSize'], \
                                    seed=0)
        
        # Chain assembler and model in a Pipeline
        pipeline = Pipeline(stages=[model])
        pipeline_model = pipeline.fit(train_data)

    # Release Cache
    train_data.unpersist()
    valid_data.unpersist()
        
    return pipeline_model