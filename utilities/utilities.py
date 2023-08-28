from imports import *
from itertools import product

##########
# SHARED #
##########

# Return the dataset with the selected features
def select_features(dataset, features, features_label, target_label):
  vectorAssembler = VectorAssembler(inputCols = features, outputCol = features_label)

  dataset = vectorAssembler.transform(dataset).select("timestamp", "id", features_label, target_label)

  return dataset

# Normalized / standardized features selection
def select_normalized_features(dataset, features, features_label, target_label):    
    # Assemble the columns into a vector column
    assembler = VectorAssembler(inputCols = features, outputCol = "raw_features")
    df_vector  = assembler.transform(dataset).select("timestamp", "id", "raw_features", target_label)

    # Create a Normalizer instance
    normalizer = Normalizer(inputCol="raw_features", outputCol=features_label)

    # Fit and transform the data
    normalized_data = normalizer.transform(df_vector).select("timestamp", "id", features_label, target_label)

    return normalized_data

'''
Description: Split and keep the original time-series order
Args:
    dataset: The dataset which needs to be splited
    proportion: A number represents the split proportion

Return: 
    train_data: The train dataset
    valid_data: The validation dataset
'''
def dataset_split(dataset, proportion):
    records_num = dataset.count()
    split_point = int(records_num * proportion)
    
    train_data = dataset.filter(dataset['id'] < split_point)
    valid_data = dataset.filter(dataset['id'] >= split_point)

    return train_data, valid_data

def show_results(results, model_name):
  trace1 = go.Scatter(
      x = results['timestamp'],
      y = results['next-market-price'].astype(float),
      mode = 'lines',
      name = 'Next Market price (usd)'
  )

  trace2 = go.Scatter(
      x = results['timestamp'],
      y = results['prediction'].astype(float),
      mode = 'lines',
      name = 'Predicted next makert price (usd)'
  )

  layout = dict(
      title= model_name + " predicitons",
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

  data = [trace1,trace2]
  fig = dict(data=data, layout=layout)
  iplot(fig, filename = model_name + " predicitons")

'''
Description: Apply calculations on Time Series Cross Validation results to form the final Model Comparison Table
Args:
    cv_result: The results from cross_validation()
    model_info: The model information which you would like to show
    evaluator_lst: The evaluator metrics which you would like to show
Return:
    comparison_df: A pandas dataset of a model on a type of Time Series Cross Validation
'''
def model_comparison(cv_result, model_info, evaluator_lst):
    # Calculate mean of all splits on chosen evaluator
    col_mean_df = cv_result[evaluator_lst].mean().to_frame().T

    # Extract model info
    model_info_df = cv_result[model_info][:1]

    # Concatenate by row
    comparison_df = pd.concat([model_info_df,col_mean_df], axis=1)
    
    return comparison_df

def model_selection(model_name, param, features_label, target_label):
    if model_name == "LinearRegression":
        model = LinearRegression(featuresCol=features_label, \
                                    labelCol=target_label, \
                                    maxIter=param['maxIter'], \
                                    regParam=param['regParam'], \
                                    elasticNetParam=param['elasticNetParam'])
        
    elif model_name == "GeneralizedLinearRegression":
        model = GeneralizedLinearRegression(featuresCol=features_label, \
                                            labelCol=target_label, \
                                            maxIter=param['maxIter'], \
                                            regParam=param['regParam'])

    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(featuresCol=features_label, \
                                        labelCol=target_label, \
                                        numTrees = param["numTrees"], \
                                        maxDepth = param["maxDepth"])

    elif model_name == "GBTRegressor":
        model = GBTRegressor(featuresCol=features_label, \
                                labelCol=target_label, \
                                maxIter = param['maxIter'], \
                                maxDepth = param['maxDepth'], \
                                stepSize = param['stepSize'])
    return model

def model_evaluation(target_label, predictions):
    mse_evaluator = RegressionEvaluator(labelCol=target_label, predictionCol="prediction", metricName='mse')
    rmse_evaluator = RegressionEvaluator(labelCol=target_label, predictionCol="prediction", metricName='rmse')
    mae_evaluator = RegressionEvaluator(labelCol=target_label, predictionCol="prediction", metricName='mae')
    r2_evaluator = RegressionEvaluator(labelCol=target_label, predictionCol="prediction", metricName='r2')

    mape = mean_absolute_percentage_error(predictions.toPandas()[target_label], predictions.toPandas()["prediction"])

    mse = mse_evaluator.evaluate(predictions)
    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    # Adjusted R-squared
    n = predictions.count()
    p = len(predictions.columns)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    results = {'rmse':rmse, 'mse':mse, 'mae':mae, 'mape':mape, 'r2':r2, 'adj_r2':adj_r2}

    return results

################
# SIMPLE MODEL #
################

# Function that create simple models (without hyperparameter tuning) and evaluate them
def model_train_valid(dataset, params, features, model_type, features_name, model_name, features_label, target_label): 
    # Select train and valid data features
    if model_type == "simple":
        dataset = select_features(dataset, features, features_label, target_label)
    elif model_type == "simple_norm" or model_type == "final_validated":
        dataset = select_normalized_features(dataset, features, features_label, target_label)

    # ALL combination of params
    param_lst = [dict(zip(params, param)) for param in product(*params.values())]

    for param in param_lst:
        # Chosen Model
        model = model_selection(model_name, param, features_label, target_label)
        
        # Split dataset
        train_data, valid_data = dataset_split(dataset, 0.9)
        # train_data,valid_data = dataset_split(dataset)


        # Chain assembler and model in a Pipeline
        pipeline = Pipeline(stages=[model])
        # Train a model and calculate running time
        start = time.time()
        pipeline_model = pipeline.fit(train_data)
        end = time.time()

        # Make predictions
        predictions = pipeline_model.transform(valid_data).select(target_label, "prediction", 'timestamp')

        # Compute validation error by several evaluators
        eval_res = model_evaluation(target_label, predictions)

        # Use dict to store each result
        results = {
            "Model": model_name,
            "Type": model_type,
            "Features": features_name,
            "Parameters": [list(param.values())],
            "RMSE": eval_res['rmse'],
            "MSE": eval_res['mse'],
            "MAE": eval_res['mae'],
            "MAPE": eval_res['mape'],
            "R2": eval_res['r2'],
            "Adjusted_R2": eval_res['adj_r2'],
            "Time": end - start,
        }

    # Transform dict to pandas dataset
    results_pd = pd.DataFrame(results, index=[0])

    return results_pd, predictions.toPandas()

#########################
# HYPERPARAMETER TUNING #
#########################

'''
Description: Use Grid Search to tune the Model 
Args:
    dataSet: The dataSet which needs to be splited
    proportion_lst: A list represents the split proportion
    feature_col: The column name of features
    label_col: The column name of label
    model_name: The module to use
    params: Parameters which want to test 
    assembler: An assembler to dataSet
    scaler: A scaler to dataSet
Return: 
    results_df: The best result in a pandas dataframe
'''
def autotuning(dataset, params, proportion_lst, features, features_name, model_name, features_label, target_label):  
    dataset = select_normalized_features(dataset, features, features_label, target_label)

    # Initialize the best result for comparison
    result_best = {"RMSE": float('inf')}
        
    # Try different proportions 
    for proportion in proportion_lst:
        # Split the dataset
        train_data,valid_data = dataset_split(dataset, proportion)
        # train_data,valid_data = dataset_split(dataset)
    
        # Cache it
        train_data.cache()
        valid_data.cache()
    
        # ALL combination of params
        param_lst = [dict(zip(params, param)) for param in product(*params.values())]
    
        for param in param_lst:
            # Chosen Model
            model = model_selection(model_name, param, features_label, target_label)
            
            # Chain assembler and model in a Pipeline
            pipeline = Pipeline(stages=[model])
            # Train a model and calculate running time
            start = time.time()
            pipeline_model = pipeline.fit(train_data)
            end = time.time()

            # Make predictions
            predictions = pipeline_model.transform(valid_data).select(target_label, "prediction", 'timestamp')

            # Compute validation error by several evaluators
            eval_res = model_evaluation(target_label, predictions)
        
            # Use dict to store each result
            results = {
                "Model": model_name,
                "Type": "autotuning",
                "Features": features_name,
                "Proportion": proportion,
                "Parameters": [list(param.values())],
                "RMSE": eval_res['rmse'],
                "MSE": eval_res['mse'],
                "MAE": eval_res['mae'],
                "MAPE": eval_res['mape'],
                "R2": eval_res['r2'],
                "Adjusted_R2": eval_res['adj_r2'],
                "Time": end - start,
            }
            
            # Only store the lowest RMSE
            if results['RMSE'] < result_best['RMSE']:
                result_best = results
                params_best = dict({key: [value] for key, value in param.items()})

        # Release Cache
        train_data.unpersist()
        valid_data.unpersist()
        
    # Transform dict to pandas dataset
    result_best_pd = pd.DataFrame(result_best)

    return result_best_pd, params_best

####################
# CROSS VALIDATION #
####################

'''
Description: Multiple Splits Cross Validation on Time Series data
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataset
'''
def multi_splits(num, n_splits):
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
        
        # Avoid to beyond the whole number of dataset
        if end > num:
            end = num
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas dataset
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Blocked Time Series Cross Validation
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataset
'''
def block_splits(num, n_splits):
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
        
    # Transform the split position list to a Pandas dataset
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Cross Validation on Time Series data
Args:
    dataset: The dataset which needs to be splited
    features_label: The column name of features
    target_label: The column name of label
    model_name: The module to use
    params: Parameters which want to test 
    assembler: An assembler to dataset
    cv_info: The type of Cross Validation
Return: 
    tsCv_df: All the splits performance of each model in a pandas dataset
'''
def cross_validation(dataset, params, cv_info, features, features_name, model_name, features_label, target_label):
    dataset = select_normalized_features(dataset, features, features_label, target_label)

    # Get the number of samples
    num = dataset.count()
    
    # Save results in a list
    results_lst = []

    # ALL combination of params
    param_lst = [dict(zip(params, param)) for param in product(*params.values())]

    for param in param_lst:
        # Chosen Model
        model = model_selection(model_name, param, features_label, target_label)

        # Identify the type of Cross Validation 
        if cv_info['cv_type'] == 'multi_splits':
            split_position_df = multi_splits(num, cv_info['splits'])
        elif cv_info['cv_type'] == 'block_splits':
            split_position_df = block_splits(num, cv_info['splits'])

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
            train_data = dataset.filter(dataset['id'].between(start, splits-1))
            valid_data = dataset.filter(dataset['id'].between(splits, end-1))

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
            predictions = pipeline_model.transform(valid_data).select(target_label, "prediction", 'timestamp')

            # Compute validation error by several evaluators
            eval_res = model_evaluation(target_label, predictions)

            # Use dict to store each result
            results = {
                "Model": model_name,
                "Type": cv_info['cv_type'],
                "Features": features_name,
                "Splits": idx + 1,
                "Train&Validation": (train_size,valid_size),                
                "Parameters": list(param.values()),
                "RMSE": eval_res['rmse'],
                "MSE": eval_res['mse'],
                "MAE": eval_res['mae'],
                "MAPE": eval_res['mape'],
                "R2": eval_res['r2'],
                "Adjusted_R2": eval_res['adj_r2'],
                "Time": end - start,
            }

            # Store each splits result
            results_lst.append(results)

            # Release Cache
            train_data.unpersist()
            valid_data.unpersist()

    # Transform dict to pandas dataset
    tsCv_pd = pd.DataFrame(results_lst)
    return tsCv_pd

#####################
# TRAIN FINAL MODEL #
#####################

def train_model(dataset, params, features, model_type, features_name, model_name, features_label, target_label):    
    dataset = select_normalized_features(dataset, features, features_label, target_label)
  
    # ALL combination of params
    param_lst = [dict(zip(params, param)) for param in product(*params.values())]
    
    for param in param_lst:
        # Chosen Model
        model = model_selection(model_name, param, features_label, target_label)
        
        # Chain assembler and model in a Pipeline
        pipeline = Pipeline(stages=[model])
        # Train a model and calculate running time
        start = time.time()
        pipeline_model = pipeline.fit(dataset)
        end = time.time()

        # Make predictions
        predictions = pipeline_model.transform(dataset).select(target_label, "prediction", 'timestamp')

        # Compute validation error by several evaluators
        eval_res = model_evaluation(target_label, predictions)

        # Use dict to store each result
        results = {
            "Model": model_name,
            "Type": model_type,
            "Features": features_name,
            "Parameters": [list(param.values())],
            "RMSE": eval_res['rmse'],
            "MSE": eval_res['mse'],
            "MAE": eval_res['mae'],
            "MAPE": eval_res['mape'],
            "R2": eval_res['r2'],
            "Adjusted_R2": eval_res['adj_r2'],
            "Time": end - start,
        }

    # Transform dict to pandas dataset
    results_pd = pd.DataFrame(results)
        
    return results_pd, pipeline_model, predictions.toPandas()