from imports import *
from config import *

#############################
# --- USEFUL PARAMETERS --- #
#############################

#Plotting parameters
colors = ['#636ffb', '#ef553b']
legend = ["Default", "Tuned"]

# Define the order for 'Splitting', 'Dataset', 'Model' columns
splitting_order = ['Block splits', 'Walk-forward splits', 'Single split']
dataset_order = ['One week', 'Fifteen days', 'One month', 'Three months']
model_order = ['LR', 'GLR', 'RF', 'GBTR']

# Mapping for models names
model_mapping = {
    "LinearRegression": "LR",
    "GeneralizedLinearRegression": "GLR",
    "RandomForestRegressor": "RF",
    "GradientBoostingTreeRegressor": "GBTR",
}

# Mapping for type names
type_mapping = {
    "default_norm": "Default",
    "default": "Default",
    "cross_val": "Tuned",
    "tuned": "Tuned",
}

# Mapping for splits names
splitting_mapping = {
    "block_splits": "Block splits",
    "walk_forward_splits": "Walk-forward splits",
    "single_split": "Single split"
}

# Mapping for datasets names
dataset_mapping = {
    "one_week": "One week",
    "fifteen_days": "Fifteen days",
    "one_month": "One month",
    "three_months": "Three months"
}


features_mapping = {
    "base_features": "Base features",
    "base_and_most_corr_features": "Base + most corr. features",
    "base_and_least_corr_features": "Base + least corr. features",
    "base_features_norm": "Base features (norm.)",
    "base_and_most_corr_features_norm": "Base + most corr. features(norm.)",
    "base_and_least_corr_features_norm": "Base + least corr. features (norm.)"
}

######################################
# --- TRAIN / VALIDATION RESULTS --- #
######################################

'''
Description: Display the dataset information
Args:
    dataset: Dataset to show
Return: None
'''
def dataset_info(dataset):
  # Print dataset
  dataset.show(20)

  # Get the number of rows
  num_rows = dataset.count()

  # Get the number of columns
  num_columns = len(dataset.columns)

  # Print the shape of the dataset
  print("Shape:", (num_rows, num_columns))

  # Print the schema of the dataset
  dataset.printSchema()

''' 
Description: Retrieves all results obtained during the train/validation phase
Args:
    splits_list: List of splitting methods used
    models_list: List of models used
    result_dir: Directory containing the results
Return:
    dataset: Dataset containing all results obtained during the train/validation phase
'''
def get_all_results(splits_list, models_list, result_dir):
    dataset = pd.DataFrame(columns=['Model', 'Type', 'Dataset', 'Splitting', 'Features', 'Parameters', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2', 'Time'])

    for split in splits_list:
        for model in models_list:
            if split == splits_list[0]: # block_splits
                dataset = pd.concat([dataset, pd.read_csv(result_dir + "/" + split + "/" + model + "_all.csv")], ignore_index=True)
            elif split == splits_list[1]: # walk_forward_splits
                dataset = pd.concat([dataset, pd.read_csv(result_dir + "/" + split + "/" + model + "_all.csv")], ignore_index=True)
            elif split == splits_list[2]: # single_split
                dataset = pd.concat([dataset, pd.read_csv(result_dir + "/" + split + "/" + model + "_all.csv")], ignore_index=True)
    
    return dataset

''' 
Description: Retrieves all the most relevant results obtained during the train/validation phase (those of the best default model and the tuned model)
Args:
    splits_list: List of splitting methods used
    models_list: List of models used
    result_dir: Directory containing the results
Return:
    dataset: Dataset containing all results obtained during the train/validation phase
'''
def get_rel_results(splits_list, models_list, result_dir):
    results = pd.DataFrame(columns=['Model', 'Type', 'Dataset', 'Splitting', 'Features', 'Parameters', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2', 'Time'])
    accuracy = pd.DataFrame(columns=['Model', 'Features', 'Splitting', 'Accuracy (default)', 'Accuracy (tuned)'])
    for split in splits_list:
        for model in models_list:
            if split == splits_list[0]: # block_splits
                results = pd.concat([results, pd.read_csv(result_dir + "/" + split + "/" + model + "_rel.csv")], ignore_index=True)
                accuracy = pd.concat([accuracy, pd.read_csv(result_dir + "/" + split + "/" + model + "_accuracy.csv")], ignore_index=True)
            elif split == splits_list[1]: # walk_forward_splits
                results = pd.concat([results, pd.read_csv(result_dir + "/" + split + "/" + model + "_rel.csv")], ignore_index=True)
                accuracy = pd.concat([accuracy, pd.read_csv(result_dir + "/" + split + "/" + model + "_accuracy.csv")], ignore_index=True)
            elif split == splits_list[2]: # single_split
                results = pd.concat([results, pd.read_csv(result_dir + "/" + split + "/" + model + "_rel.csv")], ignore_index=True)
                accuracy = pd.concat([accuracy, pd.read_csv(result_dir + "/" + split + "/" + model + "_accuracy.csv")], ignore_index=True)
    
    return results, accuracy

'''
Description: Return the dataset containing the train/validation results with the values renamed
Args:
    dataset: The dataset containing the train/validation results
    type: Type of dataset [results | accuracy]
Return: 
    dataset: Updated dataset
'''
def train_valid_dataset_fine_tuning(dataset, type):
    if type == 'results':
        # Replace results labels
        dataset['Model'] = dataset['Model'].replace(model_mapping)
        dataset['Type'] = dataset['Type'].replace(type_mapping)
        dataset['Splitting'] = dataset['Splitting'].replace(splitting_mapping)
        dataset['Features'] = dataset['Features'].replace(features_mapping)
    elif type == 'accuracy':
        # Replace accuracy labels
        dataset['Model'] = dataset['Model'].replace(model_mapping)
        dataset['Splitting'] = dataset['Splitting'].replace(splitting_mapping)
        dataset['Features'] = dataset['Features'].replace(features_mapping)

    # Convert the 'Splitting' and 'Model' columns to category type with defined order
    dataset['Splitting'] = pd.Categorical(dataset['Splitting'], categories=splitting_order, ordered=True)
    dataset['Model'] = pd.Categorical(dataset['Model'], categories=model_order, ordered=True)
    
    return dataset

'''
Description: Show the results obtained during the train / validation phase
Args:
    grouped: Grouped dataset
    colors: Colors to be used in the plot based on Type [Features | Default | Tuned]
    model: x axis for Model
    rmse: y axis for RMSE
    r2: y axis for R2
    splitting: Facet column for Splitting
    rmse_title: Title for RMSE plot
    r2_title: Title for R2 plot
Return: None
'''
def train_val_rmse_r2_plot(grouped, feature, model, rmse, r2, splitting, rmse_title, r2_title, save_path):
    # Create a bar chart for RMSE 
    fig_rmse = train_val_rmse_plot(grouped, feature, model, rmse, splitting, rmse_title)
    fig_rmse.show()

    # Create a bar chart for R2 
    fig_r2 = train_val_r2_plot(grouped, feature, model, r2, splitting, r2_title)
    fig_r2.show()

    fig_rmse.write_image(f"{save_path}train_val_rmse.png")  # Save the RMSE plot as an image
    fig_r2.write_image(f"{save_path}train_val_r2.png")  # Save the R2 plot as an image

''' Create a bar chart for RMSE  '''
def train_val_rmse_plot(grouped, feature, model, rmse, splitting, title):
    fig_rmse = px.bar(grouped, x=model, y=rmse, color=feature, facet_col=splitting, title=title)
    fig_rmse.update_layout(barmode='group')
    fig_rmse.update_layout(title_font=dict(size=24, color='black'))
    fig_rmse.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig_rmse

''' Create a bar chart for R2  '''
def train_val_r2_plot(grouped, feature, model, r2, splitting, title):
    fig_r2 = px.bar(grouped, x=model, y=r2, color=feature, facet_col=splitting, title=title)
    fig_r2.update_layout(barmode='group')
    fig_r2.update_layout(title_font=dict(size=24, color='black'))
    fig_r2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig_r2

'''
Description: Show the accuracy obtained during the train / validation phase
Args:
    grouped: Grouped dataset
    x: x axis for Model
    y1: y axis for Accuracy (default)
    y2: y axis for Accuracy (tuned)
    title: Title for the plot
Return: None    
'''
def train_val_accuracy_plot(grouped, x, y1, y2, title, save_path):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f'{splitting}' for splitting, _ in grouped])

    for i, (splitting, group) in enumerate(grouped):
        row = (i // 3) + 1
        col = (i % 3) + 1

        fig.add_trace(
            go.Bar(x=group[x], y=group[y1], name=legend[0], marker_color=colors[0], showlegend=(i==0)),
            row=row, col=col
        )

        fig.add_trace(
            go.Bar(x=group[x], y=group[y2], name=legend[1], marker_color=colors[1], showlegend=(i==0)),
            row=row, col=col
        )

        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text='Accuracy', row=row, col=col)

    fig.update_layout(title=title, showlegend=True, width=1500, height=500, title_font=dict(size=24, color='black'))
    fig.write_image(f"{save_path}train_val_accuracy.png")  # Save the accuracy plot as an image
    fig.show()

########################
# --- TEST RESULTS --- #
########################

'''
Description: Return the model parameters
Args:
    train_valid_results_raw: The dataset containing the results of the train / validation phase
    models_list: List of models
    features_list: List of features
Return:
    model_params_list: List of model parameters
'''
def get_model_parameters(train_valid_results_raw, models_list, features_list):
  # Filter train_valid_results based on Type column
  filtered_results = train_valid_results_raw[
      (train_valid_results_raw['Type'].isin(['cross_val', 'tuned'])) &
      (train_valid_results_raw['Splitting'] == 'single_split')
  ]

  model_params_list = []
  for index, row in filtered_results.iterrows():
    # Select model
    if row['Model'] == LR_MODEL_NAME:
      model = models_list[0]
    elif row['Model'] == GLR_MODEL_NAME:
      model = models_list[1]
    elif row['Model'] == RF_MODEL_NAME:
      model = models_list[2]
    elif row['Model'] == GBTR_MODEL_NAME:
      model = models_list[3]

    model_name = row['Model']
    features_label = row['Features']

    if features_label.endswith('_norm'):
      features_normalization = True
      features_label = features_label.replace("_norm", "")
    else:
      features_normalization = False

    # Select feature
    if features_label == BASE_FEATURES_LABEL:
      features = features_list[0]
    elif features_label == BASE_AND_MOST_CORR_FEATURES_LABEL:
      features = features_list[1]
    elif features_label == BASE_AND_LEAST_CORR_FEATURES_LABEL:
      features = features_list[2]

    model_params = {
        "Model_name": model_name,
        "Model": model,
        "Features_label": features_label,
        "Features": features,
        "Normalization": features_normalization
    }

    model_params_list.append(model_params)

  return model_params_list

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
Description: Return the metrics of the selected model
Args:
    target_label: The column name of target variable
    predictions: predictions made by the model
Return:
    results: Metrics obtained from the evaluation
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
Description: Evaluate final model by making predictions on the test set
Args:
    dataset: The test set to be used
    dataset_name: Name of selected dataset [one_week | fifteen_days | one_month | three_months]
    model: Trained model
    model_name: Name of the model selected
    features_normalization: Indicates whether features should be normalized or not
    features: Features to be used to make predictions
    features_name: Name of features used
    features_label: The column name of features
    target_label: The column name of target variable
Return:
    results_df: Results obtained from the evaluation
    predictions: Predictions obtained from the model
'''
def evaluate_final_model(dataset, dataset_name, model, model_name, features_normalization, features, features_name, features_label, target_label):
    # Select the type of features to be used
    dataset = select_features(dataset, features_normalization, features, features_label, target_label)

    # Make predictions
    predictions = model.transform(dataset).select(target_label, "market-price", "prediction", 'timestamp')

    # Compute validation error by several evaluators
    eval_res = model_evaluation(target_label, predictions)

    # Use dict to store each result
    results = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Features": features_name,
        "RMSE": eval_res['rmse'],
        "MSE": eval_res['mse'],
        "MAE": eval_res['mae'],
        "MAPE": eval_res['mape'],
        "R2": eval_res['r2'],
        "Adjusted_R2": eval_res['adj_r2'],
    }

    # Transform dict to pandas dataset
    results_pd = pd.DataFrame(results, index=[0])

    return results_pd, predictions

'''
Description: Evaluate final model by making predictions on the test set
Args:
    datasets_list: List of datasets
    model_params_list: List of model parameters 
Return:
    final_test_results: Results obtained from the evaluation 
    predictions_df: Predictions obtained from the model
'''
def models_testing(datasets_list, model_params_list):
  datasets_name_list = ["one_week", "fifteen_days", "one_month", "three_months"]
  predictions_df = pd.DataFrame(columns=[TARGET_LABEL, "market-price", "prediction", 'timestamp'])
  test_results = pd.DataFrame(columns=['Model', 'Dataset', 'Features', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R2', 'Adjusted_R2'])
  test_accuracy = pd.DataFrame(columns=['Model', 'Features', 'Dataset', 'Accuracy'])

  # For each model makes predictions based on the dataset type
  for model_params in model_params_list:
      for j, dataset in enumerate(datasets_list):
        model_name = model_params['Model_name']
        model = model_params['Model']
        chosen_features_label = model_params['Features_label']
        chosen_features = model_params['Features']
        features_normalization = model_params['Normalization']
        
        # Evaluate final model
        results, predictions = evaluate_final_model(dataset, datasets_name_list[j], model, model_name, features_normalization, chosen_features, chosen_features_label, FEATURES_LABEL, TARGET_LABEL)
        test_results = pd.concat([test_results, results], ignore_index=True)

        predictions = predictions.withColumn("Model", lit(model_name)).withColumn("Dataset", lit(datasets_name_list[j]))
        predictions_df = pd.concat([predictions_df, predictions.toPandas()], ignore_index=True)

        accuracy = model_accuracy(predictions)
        accuracy_data = {
            'Model': model_name,
            'Features': chosen_features_label,
            'Dataset': datasets_name_list[j],
            'Accuracy': accuracy
        }

        # Transform dict to pandas dataset
        accuracy_data_df = pd.DataFrame(accuracy_data, index=['Model'])
        test_accuracy = pd.concat([test_accuracy, accuracy_data_df], ignore_index=True)

  # Merge results and accuracy
  final_test_results = pd.merge(test_results, test_accuracy)

  return final_test_results, predictions_df

'''
Description: Return the dataset containing the test results with the values renamed
Args:
    dataset: The dataset containing the test results
Return: 
    dataset: Updated dataset
'''
def test_dataset_fine_tuning(dataset):
    # Replace results labels
    dataset['Model'] = dataset['Model'].replace(model_mapping)
    dataset['Dataset'] = dataset['Dataset'].replace(dataset_mapping)
    dataset['Features'] = dataset['Features'].replace(features_mapping)

    # Convert the 'Dataset' and 'Model' columns to category type with defined order
    dataset['Dataset'] = pd.Categorical(dataset['Dataset'], categories=dataset_order, ordered=True)
    dataset['Model'] = pd.Categorical(dataset['Model'], categories=model_order, ordered=True)

    return dataset

'''
Description: Plot the splitted datasets
Args:
    one_week: Dataset containing the data for one week
    fifteen_days: Dataset containing the data for fifteen days
    one_month: Dataset containing the data one month
    three_months: Dataset containing the data for three months
    title: Title for the plot
Return: None
'''
def show_datasets(one_week, fifteen_days, one_month, three_months, title):
  trace1 = go.Scatter(
      x = three_months['timestamp'],
      y = three_months['market-price'].astype(float),
      mode = 'lines',
      name = 'Three months market price (usd)'
  )

  trace2 = go.Scatter(
      x = one_month['timestamp'],
      y = one_month['market-price'].astype(float),
      mode = 'lines',
      name = 'One month market price (usd)'
  )

  trace3 = go.Scatter(
      x = fifteen_days['timestamp'],
      y = fifteen_days['market-price'].astype(float),
      mode = 'lines',
      name = 'Fifteen days market price (usd)'
  )

  trace4 = go.Scatter(
      x = one_week['timestamp'],
      y = one_week['market-price'].astype(float),
      mode = 'lines',
      name = 'One week market price (usd)'
  )

  layout = dict(
      title=title + " predictions",
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

  data = [trace1, trace2, trace3, trace4]
  fig = dict(data=data, layout=layout)
  iplot(fig, filename = title)

'''
Description: Plot the prediction obtained from the test phase
Args:
    dataset: The whole test dataset
    model0_name: Name of the first model
    model0_predictions: Predictions obtained from the first model
    model1_name: Name of the second model
    model1_predictions: Predictions obtained from the second model
    model2_name: Name of the third model
    model2_predictions: Predictions obtained from the third model
    model3_name: Name of the fourth model
    model3_predictions: Predictions obtained from the fourth model
    title: Title for the plot
Return: None
'''
def show_results(dataset, model0_name, model0_predictions, model1_name, model1_predictions, model2_name, model2_predictions, model3_name, model3_predictions, title):
    trace1 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset['next-market-price'].astype(float),
        mode = 'lines',
        name = 'Actual next Market price (usd)'
    )

    trace2 = go.Scatter(
        x = model0_predictions['timestamp'],
        y = model0_predictions['prediction'].astype(float),
        mode = 'lines',
        name = model0_name + ' predictions'
    )

    trace3 = go.Scatter(
        x = model1_predictions['timestamp'],
        y = model1_predictions['prediction'].astype(float),
        mode = 'lines',
        name = model1_name + ' predictions'
    )

    trace4 = go.Scatter(
        x = model2_predictions['timestamp'],
        y = model2_predictions['prediction'].astype(float),
        mode = 'lines',
        name = model2_name + ' predictions'
    )

    trace5 = go.Scatter(
        x = model3_predictions['timestamp'],
        y = model3_predictions['prediction'].astype(float),
        mode = 'lines',
        name = model3_name + ' predictions'
    )

    layout = dict(
        title=title,
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

    data = [trace1, trace2, trace3, trace4, trace5]
    fig = dict(data=data, layout=layout)
    iplot(fig, filename = title)


'''
Description: Return the accuracy of the model (how good the models are at predicting whether the price will go up or down)
Args:
    predictions: Predictions made by the model
Return: 
    accuracy: Percentage of correct predictions
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

'''
Description: Show the results obtained during the test phase
Args:
    grouped: Grouped dataset
    model: x axis for Model
    rmse: y axis for RMSE
    r2: y axis for R2
    dataset: Facet column for Dataset
    rmse_title: Title for RMSE plot
    r2_title: Title for R2 plot
Return: None
'''
def test_rmse_r2_plot(grouped, model, rmse, r2, dataset, rmse_title, r2_title, save_path):
    # Create a bar chart for RMSE 
    fig_rmse = test_rmse_plot(grouped, model, rmse, dataset, rmse_title)
    fig_rmse.show()

    # Create a bar chart for R2 
    fig_r2 = test_r2_plot(grouped, model, r2, dataset, r2_title)
    fig_r2.show()

    fig_rmse.write_image(f"{save_path}test_rmse.png")  # Save the RMSE plot as an image
    fig_r2.write_image(f"{save_path}test_r2.png")  # Save the R2 plot as an image

''' Create a bar chart for RMSE  '''
def test_rmse_plot(grouped, model, rmse, dataset, title):
    fig_rmse = px.bar(grouped, x=model, y=rmse, facet_col=dataset, title=title, color=dataset)
    fig_rmse.update_layout(barmode='group')
    fig_rmse.update_layout(title_font=dict(size=24, color='black'))
    fig_rmse.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig_rmse

''' Create a bar chart for R2  '''
def test_r2_plot(grouped, model, r2, dataset, title):
    fig_r2 = px.bar(grouped, x=model, y=r2, facet_col=dataset, title=title, color=dataset)
    fig_r2.update_layout(barmode='group')
    fig_r2.update_layout(title_font=dict(size=24, color='black'))
    fig_r2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig_r2

'''
Description: Show the accuracy obtained during the test phase
Args:
    grouped: Grouped dataset
    model: x axis for Model
    accuracy: y axis for Accuracy
    title: Title for the plot
Return: None    
'''
def test_accuracy_plot(grouped, model, accuracy, title, save_path):
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f'{dataset}' for dataset, _ in grouped])

    for i, (dataset, group) in enumerate(grouped):
        row = (i // 4) + 1
        col = (i % 4) + 1

        fig.add_trace(
            go.Bar(x=group[model], y=group[accuracy], showlegend=False),
            row=row, col=col
        )

        fig.update_xaxes(title_text=model, row=row, col=col)
        fig.update_yaxes(title_text='Accuracy', row=row, col=col)

    fig.update_layout(title=title, showlegend=True, width=1500, height=500, title_font=dict(size=24, color='black'))
    fig.write_image(f"{save_path}test_accuracy.png")  # Save the accuracy plot as an image
    fig.show()