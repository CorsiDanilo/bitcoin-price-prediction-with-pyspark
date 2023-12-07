from imports import *
from config import *

###################
# --- COMMONS --- #
###################

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
Description: Plot the division of the main dataset
Args:
    train_valid: Train / validation portion of the dataset 
    test: Test portion of the dataset 
    title: Chart title
Return: None
'''
def dataset_visualization(train_valid, test, title):
    trace1 = go.Scatter(
        x = train_valid['timestamp'],
        y = train_valid["market-price"].astype(float),
        mode = 'lines',
        name = "Train / Validation set"
    )

    trace2 = go.Scatter(
        x = test['timestamp'],
        y = test['market-price'].astype(float),
        mode = 'lines',
        name = "Test set"
    )

    layout = go.Layout(
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

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    fig.show()

'''
Description: Plot the ohlcv features
Args:
    dataset: Dataset to be considered
    features: List of ohlcv features
    title: Chart title
Return: None
'''
def ohlcv_visualization(dataset, features, title):
    trace1 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset["market-price"].astype(float),
        mode = 'lines',
        name = "Market price (usd)"
    )

    trace2 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[0][1]].astype(float),
        mode = 'lines',
        name = features[0][0]
    )

    trace3 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[1][1]].astype(float),
        mode = 'lines',
        name = features[1][0]
    )

    trace4 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[2][1]].astype(float),
        mode = 'lines',
        name = features[2][0]
    )

    trace5 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[3][1]].astype(float),
        mode = 'lines',
        name = features[3][0]
    )

    layout = go.Layout(
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
    fig = go.Figure(data=data, layout=layout)
    fig.show()

'''
Description: Plot the selected feature
Args:
    dataset: Dataset to be considered
    key: Title of the feature to be inserted in the graph
    value: Title of the feature contained in the dataset
Return: None
'''
def features_visualization(dataset, key, value):
    trace = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[value].astype(float),
        mode = 'lines',
        name = key
    )

    layout = dict(
        title=key,
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

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()

'''
Description: Plot the sma features
Args:
    dataset: Dataset to be considered
    features: List of sma features
    title: Chart title
Return: None
'''
def sma_visualization(dataset, features, title):
    trace1 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset["market-price"].astype(float),
        mode = 'lines',
        name = "Market price (usd)"
    )

    trace2 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[0][1]].astype(float),
        mode = 'lines',
        name = features[0][0]
    )

    trace3 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[1][1]].astype(float),
        mode = 'lines',
        name = features[1][0]
    )

    trace4 = go.Scatter(
        x = dataset['timestamp'],
        y = dataset[features[2][1]].astype(float),
        mode = 'lines',
        name = features[2][0]
    )

    layout = go.Layout(
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

    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)
    fig.show()