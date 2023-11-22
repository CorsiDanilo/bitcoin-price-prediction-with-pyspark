from imports import *
from plotly.subplots import make_subplots

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

def train_val_bar_plot_results(grouped, colors, x, y, title):
    fig = make_subplots(rows=3, cols=4, subplot_titles=[f'{model} ({splitting})' for ((splitting, model), _) in grouped])

    for i, ((splitting, model), group) in enumerate(grouped):
        row = (i // 4) + 1
        col = (i % 4) + 1

        fig.add_trace(
            go.Bar(x=group[x], y=group[y], name=model, marker_color=colors),
            row=row, col=col
        )

        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)

    fig.update_layout(title=title, showlegend=False, width=1500, height=1000, title_font=dict(size=24, color='black'))
    fig.show()

def train_val_bar_plot_accuracy(grouped, colors, legend, x, y1, y2, title):
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
    fig.show()

########################
# --- TEST RESULTS --- #
########################

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
Description: Evaluate final model by making predictions on the test set
Args:
    dataset: The dataSet which needs to be splited
    dataset_name: Name of selected dataset [one_week | fifteen_days | one_month | three_months]
    model: Trained model
    model_name: Model name selected
    features_normalization: Indicates whether features should be normalized (True) or not (False)
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
Description: How good the models are at predicting whether the price will go up or down
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

def test_bar_plot(grouped, colors, x, y, title):
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f'{name}' for name, _ in grouped])

    for i, (name, group) in enumerate(grouped):
        row = (i // 4) + 1
        col = (i % 4) + 1

        # Create a list of colors for each bar in the plot
        bar_colors = [colors[j%len(colors)] for j in range(len(group[x]))]

        fig.add_trace(
            go.Bar(x=group[x], y=group[y], name=name, marker_color=bar_colors, showlegend=False),
            row=row, col=col
        )

        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)

    fig.update_layout(title=title, showlegend=True, width=1500, height=500, title_font=dict(size=24, color='black'))
    fig.show()