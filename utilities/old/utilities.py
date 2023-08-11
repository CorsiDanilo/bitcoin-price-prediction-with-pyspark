# # Define the evaluation metrics
# # Notice that r2_adj metric is included when calculating r2
# metrics = ['mse', 'rmse', 'mae', 'r2']

# # ----------------------------------------------------------- #

# def simple_model(train, modelName, featureCol, labelCol):
#     if (modelName == 'LinearRegression'):
#         model = LinearRegression(featuresCol=featureCol, labelCol=labelCol)
#     elif (modelName == 'RandomForestRegressor'):
#         model = RandomForestRegressor(featuresCol=featureCol, labelCol=labelCol)
#     elif (modelName == 'GBTRegressor'):
#         model = GBTRegressor(featuresCol=featureCol, labelCol=labelCol)

#     fit_model = model.fit(train)

#     return fit_model

# # Function to compute the r2 adjusted metric
# # r2 is the r2 metric, n is the number of observations, k is the number of features
# def compute_r2adj(r2, n, k):
#   return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# # Function to evaluate a model
# def evaluate_model(predictions, modelName, typeName, label, prediction, metrics):
#   r2 = None
#   for metric in metrics:
#     evaluator = RegressionEvaluator(labelCol=label, predictionCol=prediction, metricName=metric)
#     evaluation = evaluator.evaluate(predictions)
#     print(metric.upper()+' for '+modelName+' on '+typeName+' set: '+str(evaluation))
#     if metric == 'r2':
#       print('R2_adj'+' for '+modelName+' on '+typeName+' set: '+str(compute_r2adj(evaluation, predictions.count(), len(predictions.columns))))

# # Return the dataset with the selected features
# def select_features(dataset, features, dep_var):
#   vectorAssembler = VectorAssembler(
#     inputCols = features,
#     outputCol = 'features')

#   dataset = vectorAssembler.transform(dataset)
#   dataset = dataset.select(['timestamp','index', 'features', dep_var])
#   return dataset

# # Function that create simple models (without hyperparameter tuning) and evaluate them
# def train_valid_simple_model(train_data, valid_data, modelName, features, featureCol, labelCol, metrics = ['rmse', 'r2']):
#     # Select train and valid data features
#     train_data = select_features(train_data, features, labelCol)
#     valid_data = select_features(valid_data, features, labelCol)

#     # Train the model
#     model = simple_model(train_data, modelName, featureCol, labelCol)

#     # Training set evaluation
#     training_model = model.transform(train_data)
#     evaluate_model(training_model, modelName, 'training', labelCol, 'prediction', metrics)

#     print('-----')

#     # Validation set evaluation
#     predictions_model = model.transform(valid_data)
#     evaluate_model(predictions_model, modelName, 'validation', labelCol, 'prediction', metrics)

#     return training_model.toPandas(), predictions_model.toPandas()

# # --------------------------------------------------------------------------- #

# def show_results(train, valid, training, predictions):
#   trace1 = go.Scatter(
#       x = train['timestamp'],
#       y = train['market-price'].astype(float),
#       mode = 'lines',
#       name = 'Train'
#   )

#   trace2 = go.Scatter(
#       x = valid['timestamp'],
#       y = valid['market-price'].astype(float),
#       mode = 'lines',
#       name = 'Validation'
#   )

#   trace3 = go.Scatter(
#       x = training['timestamp'],
#       y = training['prediction'].astype(float),
#       mode = 'lines',
#       name = 'Training predictions'
#   )

#   trace4 = go.Scatter(
#       x = predictions['timestamp'],
#       y = predictions['prediction'].astype(float),
#       mode = 'lines',
#       name = 'Validation predictions'
#   )

#   layout = dict(
#       title='Train, valid and prediction set with Rangeslider',
#       xaxis=dict(
#           rangeselector=dict(
#               buttons=list([
#                   #change the count to desired amount of months.
#                   dict(count=1,
#                       label='1m',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=6,
#                       label='6m',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=12,
#                       label='1y',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=36,
#                       label='3y',
#                       step='month',
#                       stepmode='backward'),
#                   dict(step='all')
#               ])
#           ),
#           rangeslider=dict(
#               visible = True
#           ),
#           type='date'
#       )
#   )

#   data = [trace1, trace2, trace3, trace4]
#   fig = dict(data=data, layout=layout)
#   iplot(fig, filename = "Train, valid and prediction set with Rangeslider")

# '''
# Description: Split and keep the original time-series order
# Args:
#     dataSet: The dataSet which needs to be splited
#     proportion: A number represents the split proportion

# Return:
#     train_data: The train dataSet
#     test_data: The test dataSet
# '''
# def trainSplit(dataSet, proportion):
#     records_num = dataSet.count()
#     split_point = int(records_num * proportion)

#     train_data = dataSet.filter(dataSet.index < split_point)
#     test_data = dataSet.filter(dataSet.index >= split_point)

#     return (train_data,test_data)

# '''
# Description: Displays the dataframes in the form of a timeseries graph
# Args:
#     train: Dataset related to training set
#     valid: Dataset related to validation set
#     training: Predictions result on the training set
#     pred: Predictions result on the validation set
# '''
# def show_results(train, valid, pred):
#   trace1 = go.Scatter(
#       x = train['timestamp'],
#       y = train['market-price'].astype(float),
#       mode = 'lines',
#       name = 'Train set'
#   )

#   trace2 = go.Scatter(
#       x = valid['timestamp'],
#       y = valid['market-price'].astype(float),
#       mode = 'lines',
#       name = 'Valid set'
#   )

#   trace3 = go.Scatter(
#       x = pred['timestamp'],
#       y = pred['prediction'].astype(float),
#       mode = 'lines',
#       name = 'Valid prediction'
#   )

#   layout = dict(
#       title='Train, valid and predictions with Rangeslider',
#       xaxis=dict(
#           rangeselector=dict(
#               buttons=list([
#                   #change the count to desired amount of months.
#                   dict(count=1,
#                       label='1m',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=6,
#                       label='6m',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=12,
#                       label='1y',
#                       step='month',
#                       stepmode='backward'),
#                   dict(count=36,
#                       label='3y',
#                       step='month',
#                       stepmode='backward'),
#                   dict(step='all')
#               ])
#           ),
#           rangeslider=dict(
#               visible = True
#           ),
#           type='date'
#       )
#   )

#   data = [trace1,trace2,trace3]
#   fig = dict(data=data, layout=layout)
#   iplot(fig, filename = "Train, valid and predictions with Rangeslider")

# '''
# Description: Use Grid Search to tune the Model
# Args:
#     dataSet: The dataSet which needs to be splited
#     proportion_lst: A list represents the split proportion
#     feature_col: The column name of features
#     label_col: The column name of label
#     ml_model: The module to use
#     params: Parameters which want to test
#     assembler: An assembler to dataSet
#     scaler: A scaler to dataSet
# Return:
#     results_df: The best result in a pandas dataframe
# '''
# def autoTuning(dataSet, proportion_lst, ml_model, feature_col, label_col, params):
#     # Initialize the best result for comparison
#     result_best = {"RMSE": float('inf')}
#     train_data_df = pd.DataFrame()
#     test_data_df = pd.DataFrame()
#     predictions_df = pd.DataFrame()

#     # Try different proportions
#     for proportion in proportion_lst:
#         # Split the dataSet
#         train_data,test_data = trainSplit(dataSet, proportion)

#         # Cache it
#         train_data.cache()
#         test_data.cache()

#         # ALL combination of params
#         param_lst = [dict(zip(params, param)) for param in product(*params.values())]

#         for param in param_lst:
#             # Chosen Model
#             if ml_model == "LinearRegression":
#                 model = LinearRegression(featuresCol=feature_col, \
#                                          labelCol=label_col, \
#                                          maxIter=param['maxIter'], \
#                                          regParam=param['regParam'], \
#                                          elasticNetParam=param['elasticNetParam'])

#             elif ml_model == "RandomForestRegressor":
#                 model = RandomForestRegressor(featuresCol=feature_col, \
#                                               labelCol=label_col, \
#                                               numTrees = param["numTrees"], \
#                                               maxDepth = param["maxDepth"])

#             elif ml_model == "GBTRegressor":
#                 model = GBTRegressor(featuresCol=feature_col, \
#                                      labelCol=label_col, \
#                                      maxIter = param['maxIter'], \
#                                      maxDepth = param['maxDepth'], \
#                                      stepSize = param['stepSize'], \
#                                      seed=0)

#             # Chain assembler and model in a Pipeline
#             pipeline = Pipeline(stages=[model])
#             # Train a model and calculate running time
#             start = time.time()
#             pipeline_model = pipeline.fit(train_data)
#             end = time.time()

#             # Make predictions
#             predictions = pipeline_model.transform(test_data)

#             # Compute test error by several evaluators
#             rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='rmse')
#             mae_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='mae')
#             r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='r2')
#             var_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='var')

#             predictions_pd = predictions.select(label_col,"prediction").toPandas()
#             mape = mean_absolute_percentage_error(predictions_pd[label_col], predictions_pd["prediction"])

#             rmse = rmse_evaluator.evaluate(predictions)
#             mae = mae_evaluator.evaluate(predictions)
#             var = var_evaluator.evaluate(predictions)
#             r2 = r2_evaluator.evaluate(predictions)
#             # Adjusted R-squared
#             n = predictions.count()
#             p = len(predictions.columns)
#             adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

#             # Use dict to store each result
#             results = {
#                 "Model": ml_model,
#                 "Proportion": proportion,
#                 "Parameters": [list(param.values())],
#                 "RMSE": rmse,
#                 "MAPE":mape,
#                 "MAE": mae,
#                 "Variance": var,
#                 "R2": r2,
#                 "Adjusted_R2": adj_r2,
#                 "Time": end - start,
#                 "Predictions": predictions.select(label_col,"prediction",'timestamp')
#             }

#             # DEBUG: show data for each split
#             show_results(train_data.toPandas(), test_data.toPandas(), predictions.toPandas())

#             # Only store the lowest RMSE
#             if results['RMSE'] < result_best['RMSE']:
#                 result_best = results

#         # Release Cache
#         train_data.unpersist()
#         test_data.unpersist()

#     # Transform dict to pandas dataframe
#     results_df = pd.DataFrame(result_best)
#     train_data_df = train_data.toPandas()
#     test_data_df = test_data.toPandas()
#     predictions_df = predictions.toPandas()

#     return results_df, train_data_df, test_data_df, predictions_df

# '''
# Description: Multiple Splits Cross Validation on Time Series data
# Args:
#     num: Number of DataSet
#     n_splits: Split times
# Return:
#     split_position_df: All set of splits position in a Pandas dataframe
# '''
# def mulTsCrossValidation(num, n_splits):
#     split_position_lst = []
#     # Calculate the split position for each time
#     for i in range(1, n_splits+1):
#         # Calculate train size and test size
#         train_size = i * num // (n_splits + 1) + num % (n_splits + 1)
#         test_size = num //(n_splits + 1)

#         # Calculate the start/split/end point for each fold
#         start = 0
#         split = train_size
#         end = train_size + test_size

#         # Avoid to beyond the whole number of dataSet
#         if end > num:
#             end = num
#         split_position_lst.append((start,split,end))

#     # Transform the split position list to a Pandas Dataframe
#     split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
#     return split_position_df

# '''
# Description: Blocked Time Series Cross Validation
# Args:
#     num: Number of DataSet
#     n_splits: Split times
# Return:
#     split_position_df: All set of splits position in a Pandas dataframe
# '''
# def blockedTsCrossValidation(num, n_splits):
#     kfold_size = num // n_splits

#     split_position_lst = []
#     # Calculate the split position for each time
#     for i in range(n_splits):
#         # Calculate the start/split/end point for each fold
#         start = i * kfold_size
#         end = start + kfold_size
#         # Manually set train-test split proportion in each fold
#         split = int(0.8 * (end - start)) + start
#         split_position_lst.append((start,split,end))

#     # Transform the split position list to a Pandas Dataframe
#     split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
#     return split_position_df

# '''
# Description: Cross Validation on Time Series data
# Args:
#     dataSet: The dataSet which needs to be splited
#     feature_col: The column name of features
#     label_col: The column name of label
#     ml_model: The module to use
#     params: Parameters which want to test
#     assembler: An assembler to dataSet
#     cv_info: The type of Cross Validation
# Return:
#     tsCv_df: All the splits performance of each model in a pandas dataframe
# '''
# def tsCrossValidation(dataSet, ml_model, feature_col, label_col, params, cv_info):
#     # Get the number of samples
#     num = dataSet.count()

#     # Save results in a list
#     result_lst = []
#     trained_models = []

#     # ALL combination of params
#     param_lst = [dict(zip(params, param)) for param in product(*params.values())]

#     for param in param_lst:
#         # Chosen Model
#         if ml_model == "LinearRegression":
#             model = LinearRegression(featuresCol=feature_col, \
#                                      labelCol=label_col, \
#                                      maxIter=param['maxIter'], \
#                                      regParam=param['regParam'], \
#                                      elasticNetParam=param['elasticNetParam'])

#         elif ml_model == "RandomForestRegressor":
#             model = RandomForestRegressor(featuresCol=feature_col, \
#                                           labelCol=label_col, \
#                                           numTrees = param["numTrees"], \
#                                           maxDepth = param["maxDepth"])

#         elif ml_model == "GBTRegression":
#             model = GBTRegressor(featuresCol=feature_col, \
#                                  labelCol=label_col, \
#                                  maxIter = param['maxIter'], \
#                                  maxDepth = param['maxDepth'], \
#                                  stepSize = param['stepSize'])

#         # Identify the type of Cross Validation
#         if cv_info['cv_type'] == 'mulTs':
#             split_position_df = mulTsCrossValidation(num, cv_info['kSplits'])
#         elif cv_info['cv_type'] == 'blkTs':
#             split_position_df = blockedTsCrossValidation(num, cv_info['kSplits'])

#         for position in split_position_df.itertuples():
#             # Get the start/split/end position from a kind of Time Series Cross Validation
#             start = getattr(position, 'start')
#             splits = getattr(position, 'split')
#             end = getattr(position, 'end')
#             idx  = getattr(position, 'Index')

#             # Train/Test size
#             train_size = splits - start
#             test_size = end - splits

#             # Get training data and test data
#             train_data = dataSet.filter(dataSet.index.between(start, splits-1))
#             test_data = dataSet.filter(dataSet.index.between(splits, end-1))

#             # Cache it
#             train_data.cache()
#             test_data.cache()

#             # Chain assembler and model in a Pipeline
#             pipeline = Pipeline(stages=[model])
#             # Train a model and calculate running time
#             start = time.time()
#             pipeline_model = pipeline.fit(train_data)
#             end = time.time()

#             # Make predictions
#             predictions = pipeline_model.transform(test_data)

#             # Compute test error by several evaluator
#             rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='rmse')
#             mae_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='mae')
#             r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='r2')
#             var_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName='var')

#             predictions_pd = predictions.select(label_col,"prediction").toPandas()
#             mape = mean_absolute_percentage_error(predictions_pd[label_col], predictions_pd["prediction"])

#             rmse = rmse_evaluator.evaluate(predictions)
#             mae = mae_evaluator.evaluate(predictions)
#             var = var_evaluator.evaluate(predictions)
#             r2 = r2_evaluator.evaluate(predictions)
#             # Adjusted R-squared
#             n = predictions.count()
#             p = len(predictions.columns)
#             adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

#             # Use dict to store each result
#             results = {
#                 "Model": ml_model,
#                 'CV_type': cv_info['cv_type'],
#                 "Splits": idx + 1,
#                 "Train&Test": (train_size,test_size),
#                 "Parameters": list(param.values()),
#                 "RMSE": rmse,
#                 "MAPE": mape,
#                 "MAE": mae,
#                 "Variance": var,
#                 "R2": r2,
#                 "Adjusted_R2": adj_r2,
#                 "Time": end - start
#             }

#             # DEBUG: show data for each split
#             show_results(train_data.toPandas(), test_data.toPandas(), predictions.toPandas())

#             # Store each splits result
#             result_lst.append(results)

#             # Append the trained model to the list
#             trained_models.append(pipeline_model)

#             # Release Cache
#             train_data.unpersist()
#             test_data.unpersist()

#     # Transform dict to pandas dataframe
#     tsCv_df = pd.DataFrame(result_lst)
#     return tsCv_df, trained_models