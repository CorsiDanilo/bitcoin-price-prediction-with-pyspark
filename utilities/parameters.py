from imports import *

# For reproducibility
RANDOM_SEED = 42

'''
Description: Returns the  parameters of the selected splitting type
Args:
    split_type: Type of splitting [block_splits | walk_forward_splits |single_split]
Return: 
    params: Parameters of the selected splitting type
'''
def get_splitting_params(split_type):
    if split_type == "block_splits":
        # Block splits time series
        params = {'split_type':'block_splits',
                  'splits': 5}
    elif split_type == "walk_forward_splits":
        # Walk forward splits time series
        params = {'split_type':'walk_forward_splits',
                  'min_obser': 20000,
                  'sliding_window': 5000}
    elif split_type == "single_split":
        # Single split time series
        params = {'split_type':'single_split',
                  'split_label':'months',
                  'split_value': 1}

    return params

'''
Description: Returns the default parameters of the selected model
Args:
    model_name: Name of the selected model [LinearRegression | GeneralizedLinearRegression | RandomForestRegressor | GradientBoostingTreeRegressor]
Return: 
    params: Parameters list of the selected model
'''
def get_defaults_model_params(model_name):
    if (model_name == 'LinearRegression'):
        params = {
            'maxIter' : [100],
            'regParam' : [0.0],
            'elasticNetParam' : [0.0]
        }   
    if (model_name == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [25],
            'regParam' : [0]
        }
    elif (model_name == 'RandomForestRegressor'):
        params = {
            'numTrees' : [20],
            'maxDepth' : [5],
            'seed' : [RANDOM_SEED]
            }
    elif (model_name == 'GradientBoostingTreeRegressor'):
        params = {
            'maxIter' : [20],
            'maxDepth' : [5],
            'stepSize': [0.1],
            'seed' : [RANDOM_SEED]
        }
    
    return params

'''
Description: Returns the model grid parameters of the selected model
Args:
    model_name: Name of the selected model [LinearRegression | GeneralizedLinearRegression | RandomForestRegressor | GradientBoostingTreeRegressor]
Return: 
    params: Parameters list of the selected model
'''
def get_model_grid_params(model_name):
    if (model_name == 'LinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80, 100],
            'regParam' : np.arange(0,1,0.2).round(decimals=2),
            'elasticNetParam' : np.arange(0,1,0.2).round(decimals=2)
        }
    if (model_name == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80],
            'regParam' : [0, 0.1, 0.2],
            'family': ['gaussian', 'gamma'],
            'link': ['log', 'identity', 'inverse']
        }
    elif (model_name == 'RandomForestRegressor'):
        params = {
            'numTrees' : [3, 5, 10, 20, 30],
            'maxDepth' : [3, 5, 10],
            'seed' : [RANDOM_SEED]
        }
    elif (model_name == 'GradientBoostingTreeRegressor'):
        params = {
            'maxIter' : [3, 5, 10, 20, 30],
            'maxDepth' : [3, 5, 10],
            'stepSize': [0.1, 0.3, 0.5, 0.7],
            'seed' : [RANDOM_SEED]
        }

    return params

'''
Description: Choose the best model parameters
Args:
    parameters: Input parameters
Return: 
    grouped_scores: Scores of all parameters
    best_params: Parameters with the best score
'''
def choose_best_params(parameters):
    # Calculate the weight of each value in the "Splits" column
    parameters['Split weight'] = parameters['Splits'].rank(ascending=True)

    # Normalize the split weights to a scale of 0 to 1
    parameters['Split weight'] = parameters['Split weight'] / parameters['Split weight'].max()
    
    # Normalize the RMSE values to a scale of 0 to 1
    rmse_weight = 1 - (parameters['RMSE'] / parameters['RMSE'].max())

    # Add the RMSE weight as a new column
    parameters['RMSE weight'] = rmse_weight
    
    # To access the values of a tuple 
    parameters['Parameters'] = parameters['Parameters'].apply(tuple)

    # Calculate the frequency of each unique value in the "Parameters" column
    freq = parameters['Parameters'].value_counts(normalize=True)

    # Normalize the frequencies to a scale of 0 to 1
    freq_norm = freq / freq.max()

    # Create a new column called "Frequency" with the normalized frequencies
    parameters['Frequency weight'] = parameters['Parameters'].map(freq_norm)

    # Group the rows by the "Parameters" column and calculate the average of the other columns
    grouped_scores = parameters[['Parameters', 'Split weight', 'RMSE weight', "Frequency weight"]].groupby('Parameters').mean()
    
    # Calculate the final score for each row
    grouped_scores['Final score'] = grouped_scores['Frequency weight'] *  grouped_scores['Split weight'] * grouped_scores['RMSE weight']

    # Sort the DataFrame by the "Final score" column in descending order
    grouped_scores = grouped_scores.sort_values(by='Final score', ascending=False)

    # Get the index of the row with the highest final score
    best_params = grouped_scores['Final score'].idxmax()

    return grouped_scores, best_params

'''
Description: Returns the best model parameters
Args:
    parameters: Parameters to be entered in the parameter grid of the selected model
    model_name: Name of the selected model [LinearRegression | GeneralizedLinearRegression | RandomForestRegressor | GradientBoostingTreeRegressor]
Return: 
    params: Parameters list of the selected model
'''
def get_best_model_params(parameters, model_name):
    if (model_name == 'LinearRegression'):
        params = {
            'maxIter' : [parameters[0]],
            'regParam' : [parameters[1]],
            'elasticNetParam' : [parameters[2]]
        }   
    if (model_name == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [parameters[0]],
            'regParam' : [parameters[1]],
            'family': [parameters[2]],
            'link': [parameters[3]]
        }
    elif (model_name == 'RandomForestRegressor'):
        params = {
            'numTrees' : [parameters[0]],
            'maxDepth' : [parameters[1]],
            'seed' : [parameters[2]]
            }
    elif (model_name == 'GradientBoostingTreeRegressor'):
        params = {
            'maxIter' : [parameters[0]],
            'maxDepth' : [parameters[1]],
            'stepSize': [parameters[2]],
            'seed' : [parameters[3]]
        }
        
    return params