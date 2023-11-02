from imports import *

###################
# --- COMMONS --- #
###################

'''
Description: Return the dataset with the selected features
Args:
    dataset: The dataset from which to extract the features
    features_normalization: Indicates whether features should be normalized (True) or not (False)
    features: list of features to be extracted
    features_label: The column name of features
    target_label: The column name of target variable
Return: 
    dataset: Dataset with the selected features
'''
def select_features(dataset, features_normalization, features, features_label, target_label):
    if features_normalization:
        # Assemble the columns into a vector column
        assembler = VectorAssembler(inputCols = features, outputCol = "raw_features")
        df_vector  = assembler.transform(dataset).select("timestamp", "id", "market-price", "raw_features", target_label)

        # Create a Normalizer instance
        normalizer = Normalizer(inputCol="raw_features", outputCol=features_label)

        # Fit and transform the data
        dataset = normalizer.transform(df_vector).select("timestamp", "id", "market-price", features_label, target_label)
    else:
        # Assemble the columns into a vector column
        assembler = VectorAssembler(inputCols = features, outputCol = features_label)
        dataset = assembler.transform(dataset).select("timestamp", "id", "market-price", features_label, target_label)

    return dataset

'''
Description: Plot the results obtained
Args:
    results: Results to be displayed
    title: Chart title
Return: None
'''
def show_results(dataset, train, valid, title, onlyTrain):
    if not onlyTrain:     
        trace1 = go.Scatter(
            x = dataset['timestamp'],
            y = dataset['next-market-price'].astype(float),
            mode = 'lines',
            name = 'Next Market price (usd)'
        )

        trace2 = go.Scatter(
            x = train['timestamp'],
            y = train['prediction'].astype(float),
            mode = 'lines',
            name = '(Train) Predicted next makert price (usd)'
        )

        trace3 = go.Scatter(
            x = valid['timestamp'],
            y = valid['prediction'].astype(float),
            mode = 'lines',
            name = '(Valid) Next Market price (usd)'
        )
    else:
        trace1 = go.Scatter(
            x = train['timestamp'],
            y = train['next-market-price'].astype(float),
            mode = 'lines',
            name = 'Next Market price (usd)'
        )

        trace2 = go.Scatter(
            x = train['timestamp'],
            y = train['prediction'].astype(float),
            mode = 'lines',
            name = 'Predicted next makert price (usd)'
        )
        
    layout = dict(
        title= title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    # Change the count to desired amount of months.
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

    if not onlyTrain:     
        data = [trace1,trace2,trace3]
    else:
        data = [trace1,trace2]

    fig = dict(data=data, layout=layout)
    iplot(fig, filename = title)
   
'''
Description: Returns the average of the results obtained
Args:
    results: Obtained results from the model
    model_info: The model information to show
    evaluator_lst: The evaluator metrics to show
Return:
    comparison_df: Average of the results in a Pandas dataframe
'''
def model_comparison(results, model_info, evaluator_lst):
    # Calculate mean of all results
    col_mean_df = results[evaluator_lst].mean().to_frame().T

    # Extract model info
    model_info_df = results[model_info][:1]

    # Concatenate by row
    comparison_df = pd.concat([model_info_df, col_mean_df], axis=1)
    
    return comparison_df

'''
Description: Selection of the model to be initialized
Args:
    model_name: Model name selected
    param: Parameters of the selected model
    features_label: The column name of features
    target_label: The column name of target variable
Return:
    model: Initialized model
'''
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
                                        maxDepth = param["maxDepth"], \
                                        seed=param['seed'])

    elif model_name == "GradientBoostingTreeRegressor":
        model = GBTRegressor(featuresCol=features_label, \
                                labelCol=target_label, \
                                maxIter = param['maxIter'], \
                                maxDepth = param['maxDepth'], \
                                stepSize = param['stepSize'], \
                                seed=param['seed'])

    return model

'''
Description: Evaluation of the selected model
Args:
    target_label: The column name of target variable
    predictions: predictions made by the model
Return:
    results: Results obtained from the evaluation
'''
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

'''
Description: Tell how good the models are at predicting whether the price will go up or down
Args:
    dataset: The dataset which needs to be splited
Return: 
    accuracy: Return the percentage of correct predictions
'''
def model_accuracy(dataset):    
    # Compute the number of total rows in the DataFrame.
    total_rows = dataset.count()

    # Create a column "correct_prediction" which is worth 1 if the prediction is correct, otherwise 0
    dataset = dataset.withColumn(
        "correct_prediction",
        (
            (col("market-price") < col("next-market-price")) & (col("market-price") < col("prediction"))
        ) | (
            (col("market-price") > col("next-market-price")) & (col("market-price") > col("prediction"))
        )
    )

    # Count the number of correct predictions
    correct_predictions = dataset.filter(col("correct_prediction")).count()

    # Compite percentage of correct predictions
    accuracy = (correct_predictions / total_rows) * 100
        
    return accuracy

###########################
# --- MULTIPLE SPLITS --- #
###########################

'''
Description: Block splits time series split
Args:
    num: Number of datasets
    n_splits: Split times
Return: 
    split_position_df: All sets of split positions in a Pandas dataset.
'''
def block_splits(num, n_splits):
    # Calculate the split position for each fold 
    kfold_size = num // n_splits
    split_position_lst = []
    for i in range(n_splits):
        # Calculate the start/split/end point for each fold
        start = i * kfold_size
        end = start + kfold_size

        # Manually set train-validation split proportion in each fold
        split = int(0.8 * (end - start)) + start
        split_position_lst.append((start, split,end))
        
    # Transform the split position list to a Pandas dataset
    split_position_df = pd.DataFrame(split_position_lst, columns=['start', 'split', 'end'])

    return split_position_df

'''
Description: Walk forward time series split
Args:
    num: Number of dataset
    min_obser: Minimum number of observations
    sliding_window: Sliding Window
Return: 
    split_position_df: All sets of split positions in a Pandas dataset.
''' 
def walk_forward_splits(num, min_obser, sliding_window):
    # Calculate the split position for each fold 
    split_positions = []
    start = 0
    while start + min_obser + sliding_window <= num:
        split_positions.append((start, start + min_obser, start + min_obser + sliding_window))
        start += sliding_window

    # Transform the split position list to a Pandas dataset
    split_position_df = pd.DataFrame(split_positions, columns=['start', 'split', 'end'])

    return split_position_df

'''
Description: Perform train / validation using multiple splitting methods
Args:
    dataset: The dataset which needs to be splited
    params: Parameters which want to test 
    splitting_info: The splitting type method [block_splits | walk_forward_splits]
    model_name: Model name selected
    model_type: Model type [default | default_norm | hyp_tuning | cross_val]
    features_normalization: Indicates whether features should be normalized or not
    features: Features to be used to make predictions
    features_name: Name of features used
    features_label: The column name of features
    target_label: The column name of target variable
Return: 
    train_results_df: All the train splits performances in a pandas dataset
    valid_results_df: All the validations splits performances in a pandas dataset
    train_predictions_df: All the train splits predictions in a pandas dataset
    valid_predictions_df: All the validations splits predictions in a pandas dataset
'''
def multiple_splits(dataset, params, splitting_info, model_name, model_type, features_normalization, features, features_name, features_label, target_label):
    # Select the type of features to be used
    dataset = select_features(dataset, features_normalization, features, features_label, target_label)

    # Plot to show
    current_plot = 0
    next_plot = 1

    # Get the number of samples
    num = dataset.count()

    # Save results in a list
    all_train_results = []
    all_valid_results = []
    best_split_result = []

    # Initialize an empty list to store predictions
    all_train_predictions = []  
    all_valid_predictions = [] 

    # Identify the splitting type
    if splitting_info['split_type'] == 'block_splits':
        split_position_df = block_splits(num, splitting_info['splits'])
    elif splitting_info['split_type'] == 'walk_forward_splits':
        split_position_df = walk_forward_splits(num, splitting_info['min_obser'], splitting_info['sliding_window'])

    for position in split_position_df.itertuples():
        best_result = {"RMSE": float('inf')}

        # Get the start/split/end position based on the splitting type
        start = getattr(position, 'start')
        splits = getattr(position, 'split')
        end = getattr(position, 'end')
        idx  = getattr(position, 'Index')
        
        # Train / validation size
        train_size = splits - start
        valid_size = end - splits

        # Get training data and validation data
        train_data = dataset.filter(dataset['id'].between(start, splits-1))
        valid_data = dataset.filter(dataset['id'].between(splits, end-1))

        # Cache them
        train_data.cache()
        valid_data.cache()
        
        # All combination of params
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
            train_predictions = pipeline_model.transform(train_data).select(target_label, "market-price", "prediction", 'timestamp')
            valid_predictions = pipeline_model.transform(valid_data).select(target_label, "market-price", "prediction", 'timestamp')

            # Show plots
            if (model_type != "hyp_tuning") and (current_plot == 1 or current_plot % 5 == 0 or next_plot % 5 == 0 or splitting_info['split_type'] == "block_splits"):
                show_results(dataset.toPandas(), train_predictions.toPandas(), valid_predictions.toPandas(), model_name + " predictions on split " +  str(idx + 1), False)            
            current_plot = current_plot + 1
            next_plot = next_plot + 1

            if model_type == "default" or model_type == "default_norm" or model_type == "cross_val":
                # Append predictions to the list
                all_train_predictions.append(train_predictions) 
                all_valid_predictions.append(valid_predictions)

            # Compute validation error by several evaluators
            train_eval_res = model_evaluation(target_label, train_predictions)
            valid_eval_res = model_evaluation(target_label, valid_predictions)

            # Use dict to store each result
            train_results = {
                "Model": model_name,
                "Type": model_type,
                "Dataset": 'train',
                "Splitting": splitting_info['split_type'],
                "Features": features_name,
                "Splits": idx + 1,
                "Train / Validation": (train_size,valid_size),                
                "Parameters": list(param.values()),
                "RMSE": train_eval_res['rmse'],
                "MSE": train_eval_res['mse'],
                "MAE": train_eval_res['mae'],
                "MAPE": train_eval_res['mape'],
                "R2": train_eval_res['r2'],
                "Adjusted_R2": train_eval_res['adj_r2'],
                "Time": end - start,
            }

            valid_results = {
                "Model": model_name,
                "Type": model_type,
                "Dataset": 'valid',
                "Splitting": splitting_info['split_type'],
                "Features": features_name,
                "Splits": idx + 1,
                "Train / Validation": (train_size,valid_size),                
                "Parameters": list(param.values()),
                "RMSE": valid_eval_res['rmse'],
                "MSE": valid_eval_res['mse'],
                "MAE": valid_eval_res['mae'],
                "MAPE": valid_eval_res['mape'],
                "R2": valid_eval_res['r2'],
                "Adjusted_R2": valid_eval_res['adj_r2'],
                "Time": end - start,
            }

            if model_type == "hyp_tuning":
                # Store the result with the lowest RMSE and the associated parameters
                if valid_results['RMSE'] < best_result['RMSE']:
                    best_result = valid_results

            if model_type == "default" or model_type == "default_norm" or model_type == "cross_val":
                # Store results for each split
                all_train_results.append(train_results)
                all_valid_results.append(valid_results)
        
        # Release Cache
        train_data.unpersist()
        valid_data.unpersist()

        if model_type == "hyp_tuning":
            # Store the best result for each split
            best_split_result.append(best_result) 
            print("Best parameters chosen for split " + str(idx + 1) + ": " + str(best_result["Parameters"]))

    if model_type == "hyp_tuning":
        # Transform dict to pandas dataset
        best_split_result_df = pd.DataFrame(best_split_result)

        return best_split_result_df
    
    if model_type == "default" or model_type == "default_norm" or model_type == "cross_val":
        # Transform dict to pandas dataset
        all_train_results_df = pd.DataFrame(all_train_results)
        all_valid_results_df = pd.DataFrame(all_valid_results)

        # Iterate for each train and validation predictions and concatenate it with the final one
        all_train_predictions_df = pd.DataFrame()
        for pred in all_train_predictions:
            all_train_predictions_df = pd.concat([all_train_predictions_df, pred.select("*").toPandas()], ignore_index=True)

        all_valid_predictions_df = pd.DataFrame()
        for pred in all_valid_predictions:
            all_valid_predictions_df = pd.concat([all_valid_predictions_df, pred.select("*").toPandas()], ignore_index=True)

        return all_train_results_df, all_valid_results_df, all_train_predictions_df, all_valid_predictions_df

########################
# --- SINGLE SPLIT --- #
########################

'''
Description: Short term time series split
Args:
    dataset: The dataset which needs to be splited
    label: Type of splitting [weeks | months | years]
    proportion: A number represents the split proportion
Return: 
    train_data: The train dataset
    valid_data: The valid dataset
'''
def short_term_split(dataset, split_label, split_value):
    # Retrieve the last timestamp value
    last_value = dataset.agg(last("timestamp")).collect()[0][0]

    # Subtract the value from the last timestamp based on the split label
    match split_label:
        case "weeks":
            split_date = last_value - relativedelta(weeks=split_value)
        case "months":
            split_date = last_value - relativedelta(months=split_value)
        case "years":
            split_date = last_value - relativedelta(years=split_value)
        case _:
            return 

    # Split the dataset based on the desired date
    train_data = dataset[dataset['timestamp'] <= split_date]
    test_df = dataset[dataset['timestamp'] > split_date]

    return train_data, test_df

'''
Description: Perform train / validation using single split method
Args:
    dataset: The dataset which needs to be splited
    params: Parameters which want to test 
    splitting_info: The splitting type method [short_term_split]
    model_name: Model name selected
    model_type: Model type [default | default_norm | final_validated]
    features_normalization: Indicates whether features should be normalized or not
    features: Features to be used to make predictions
    features_name: Name of features used
    features_label: The column name of features
    target_label: The column name of target variable
Return: 
    train_results_df: All the train splits performances in a pandas dataset
    valid_results_df: All the validations splits performances in a pandas dataset
    train_predictions_df: All the train splits predictions in a pandas dataset
    valid_predictions_df: All the validations splits predictions in a pandas dataset
'''
def single_split(dataset, params, splitting_info, model_name, model_type, features_normalization, features, features_name, features_label, target_label):
    # Select the type of features to be used
    dataset = select_features(dataset, features_normalization, features, features_label, target_label)

    # Get the number of samples
    num = dataset.count()

    # Get training data and validation data
    train_data, valid_data = short_term_split(dataset, splitting_info['split_label'], splitting_info['split_value'])
    
    # Train / validation size
    train_size = train_data.count()
    valid_size = valid_data.count()

    # Cache them
    train_data.cache()
    valid_data.cache()
    
    # All combination of params
    param_lst = [dict(zip(params, param)) for param in product(*params.values())]

    for param in param_lst:
        best_result = {"RMSE": float('inf')}

        # Chosen Model
        model = model_selection(model_name, param, features_label, target_label)

        # Chain assembler and model in a Pipeline
        pipeline = Pipeline(stages=[model])

        # Train a model and calculate running time
        start = time.time()
        pipeline_model = pipeline.fit(train_data)
        end = time.time()

        # Make predictions
        train_predictions = pipeline_model.transform(train_data).select(target_label, "market-price", "prediction", 'timestamp')
        valid_predictions = pipeline_model.transform(valid_data).select(target_label, "market-price", "prediction", 'timestamp')
        
        # Show plots
        show_results(dataset.toPandas(), train_predictions.toPandas(), valid_predictions.toPandas(), model_name + " predictions", False)

        # Compute validation error by several evaluators
        train_eval_res = model_evaluation(target_label, train_predictions)
        valid_eval_res = model_evaluation(target_label, valid_predictions)

        # Use dict to store each result
        train_results = {
            "Model": model_name,
            "Type": model_type,
            "Dataset": 'train',
            "Splitting": splitting_info['split_type'],
            "Features": features_name,
            "Train / Validation": (train_size,valid_size),                
            "Parameters": list(param.values()),
            "RMSE": train_eval_res['rmse'],
            "MSE": train_eval_res['mse'],
            "MAE": train_eval_res['mae'],
            "MAPE": train_eval_res['mape'],
            "R2": train_eval_res['r2'],
            "Adjusted_R2": train_eval_res['adj_r2'],
            "Time": end - start,
        }

        valid_results = {
            "Model": model_name,
            "Type": model_type,
            "Dataset": 'valid',
            "Splitting": splitting_info['split_type'],
            "Features": features_name,
            "Train / Validation": (train_size,valid_size),                
            "Parameters": list(param.values()),
            "RMSE": valid_eval_res['rmse'],
            "MSE": valid_eval_res['mse'],
            "MAE": valid_eval_res['mae'],
            "MAPE": valid_eval_res['mape'],
            "R2": valid_eval_res['r2'],
            "Adjusted_R2": valid_eval_res['adj_r2'],
            "Time": end - start,
        }
        
    # Release Cache
    train_data.unpersist()
    valid_data.unpersist()

    # Store train and validation results into pandas dataset
    train_results_df = pd.DataFrame.from_dict(train_results, orient='index').T
    valid_results_df = pd.DataFrame.from_dict(valid_results, orient='index').T

    return train_results_df, valid_results_df, train_predictions.toPandas(), valid_predictions.toPandas()
        
'''
Description: Evaluation of the final trained model
Args:
    dataset: The dataset which needs to be splited
    params: Parameters which want to test 
    model_name: Model name selected
    model_type: Model type [final_trained]
    features_normalization: Indicates whether features should be normalized or not
    features: Features to be used to make predictions
    features_name: Name of features used
    features_label: The column name of features
    target_label: The column name of target variable
Return: 
    results_df: Results obtained from the evaluation
    pipeline_model: Final trained model
    predictions: Predictions obtained from the model
'''
def evaluate_trained_model(dataset, params, model_name, model_type, features_normalization, features, features_name, features_label, target_label):    
    # Select the type of features to be used
    dataset = select_features(dataset, features_normalization, features, features_label, target_label)
  
    # All combination of params
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
        predictions = pipeline_model.transform(dataset).select(target_label, "market-price", "prediction", 'timestamp')

        # Compute validation error by several evaluators
        eval_res = model_evaluation(target_label, predictions)

        #  Use dict to store each result
        results = {
            "Model": model_name,
            "Type": model_type,
            "Dataset": 'train',
            "Splitting": "whole_train_valid",
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
    results_df = pd.DataFrame(results)

    # Show plots
    show_results(None, predictions.toPandas(), None, model_name + " prediction on the whole train / validation set", True)
        
    return results_df, pipeline_model, predictions.toPandas()