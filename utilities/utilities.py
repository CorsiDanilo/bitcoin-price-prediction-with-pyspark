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

# ----------------------------------------------------------- #

def simple_model(train, model,featureCol, labelCol):
    if (model == 'LinearRegression'):
        lr = LinearRegression(featuresCol=featureCol, labelCol=labelCol)
        lr_model = lr.fit(train)

        return lr_model
    elif (model == 'RandomForestRegressor'):
        rf = RandomForestRegressor(featuresCol=featureCol, labelCol=labelCol)
        rf_model = rf.fit(train)

        return rf_model
    elif (model == 'GBTRegressor'):
        gbt = GBTRegressor(featuresCol=featureCol, labelCol=labelCol)
        gbt_model = gbt.fit(train)

        return gbt_model

# Function to compute the r2 adjusted metric
# r2 is the r2 metric, n is the number of observations, k is the number of features
def compute_r2adj(r2, n, k):
  return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# Function to evaluate a model
def evaluate_models(predictions, modelName, typeName, label, prediction, metrics):
  r2 = None
  for metric in metrics:
    evaluator = RegressionEvaluator(labelCol=label, predictionCol=prediction, metricName=metric)
    evaluation = evaluator.evaluate(predictions)
    print(metric.upper()+' for '+modelName+' on '+typeName+' set: '+str(evaluation))
    if metric == 'r2':
      print('R2_adj'+' for '+modelName+' on '+typeName+' set: '+str(compute_r2adj(evaluation, predictions.count(), len(predictions.columns))))

# Return the dataset with the selected features
def select_features(dataset, features, dep_var):
  vectorAssembler = VectorAssembler(
    inputCols = features,
    outputCol = 'features')

  dataset = vectorAssembler.transform(dataset)
  dataset = dataset.select(['timestamp','index', 'features', dep_var])
  return dataset

# Function that create simple models (without hyperparameter tuning) and evaluate them
def test_best_features(train_data, valid_data, model, features, featureCol, labelCol, metrics = ['rmse', 'r2']):
    train_data = select_features(train_data, features, labelCol)
    valid_data = select_features(valid_data, features, labelCol)

    if (model == 'LinearRegression'):
        # Train the models
        lr = simple_linear_regression_model(train_data, featureCol, labelCol)

        # Training set evaluation
        lr_training = lr.transform(train_data)
        evaluate_models(lr_training, model, 'training', labelCol, 'prediction', metrics)

        print('-----')

        # Validation set evaluation
        lr_predictions = lr.transform(valid_data)
        evaluate_models(lr_predictions, model, 'validation', labelCol, 'prediction', metrics)

        return lr_training, lr_predictions
    elif (model == 'RandomForestRegressor'):
        # Train the models
        rf = simple_random_forest_model(train_data, featureCol, labelCol)

        # Training set evaluation
        rf_training = rf.transform(train_data)
        evaluate_models(rf_training, model, 'training', labelCol, 'prediction', metrics)

        print('-----')

        # Validation set evaluation
        rf_predictions = rf.transform(valid_data)
        evaluate_models(rf_predictions, model, 'validation', labelCol, 'prediction', metrics)
    
        return rf_training, rf_predictions
    elif (model == 'GBTRegressor'):
        # Train the models
        gbt = simple_gbt_model(train_data, featureCol, labelCol)

        # Training set evaluation
        gbt_training = gbt.transform(train_data)
        evaluate_models(gbt_training, model, 'training', labelCol, 'prediction', metrics)
    
        print('-----')

        # Validation set evaluation
        gbt_predictions = gbt.transform(valid_data)
        evaluate_models(gbt_predictions, model, 'validation', labelCol, 'prediction', metrics)
        
        return gbt_training, gbt_predictions

# --------------------------------------------------------------------------- # 

# Hyperparameter tuning for the model
def random_forest_cross_val(dataset, dep_var, param_grid,k_fold=5):
    rf = RandomForestRegressor(featuresCol='features', labelCol=dep_var)
    pipeline = Pipeline(stages=[rf])

    # Default (too much memory!!)
    # param_grid = ParamGridBuilder()\
    # .addGrid(rf.maxDepth, [8, 9, 10]) \
    # .addGrid(rf.numTrees, [40, 60, 80]) \
    # .build()

    cross_val = CrossValidator(estimator=pipeline,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol=dep_var),
                               numFolds=k_fold,
                               collectSubModels=True
                               )

    # Run cross-validation, and choose the best set of parameters.
    cv_model = cross_val.fit(dataset)

    return cv_model

# Summarizes all the models trained during cross validation
def summarize_rf_models(cv_models):
    for k, models in enumerate(cv_models):
        print("*************** Fold #{:d} ***************\n".format(k+1))
        for i, m in enumerate(models):
            print("--- Model #{:d} out of {:d} ---".format(i+1, len(models)))
            print("\tParameters: maxDepth=[{:.3f}]; numTrees=[{:.3f}] ".format(m.stages[-1]._java_obj.getMaxDepth(), m.stages[-1]._java_obj.getNumTrees()))
            print("\tModel summary: {}\n".format(m.stages[-1]))
        print("***************************************\n")

# --------------------------------------------------------------------------- # 

def show_results(train, valid, training, predictions):
  trace1 = go.Scatter(
      x = train['timestamp'],
      y = train['market-price'].astype(float),
      mode = 'lines',
      name = 'Train'
  )

  trace2 = go.Scatter(
      x = valid['timestamp'],
      y = valid['market-price'].astype(float),
      mode = 'lines',
      name = 'Validation'
  )

  trace3 = go.Scatter(
      x = training['timestamp'],
      y = training['prediction'].astype(float),
      mode = 'lines',
      name = 'Training predictions'
  )

  trace4 = go.Scatter(
      x = predictions['timestamp'],
      y = predictions['prediction'].astype(float),
      mode = 'lines',
      name = 'Validation predictions'
  )

  layout = dict(
      title='Train, valid and prediction set with Rangeslider',
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

  data = [trace1, trace2, trace3, trace4]
  fig = dict(data=data, layout=layout)
  iplot(fig, filename = "Train, valid and prediction set with Rangeslider")