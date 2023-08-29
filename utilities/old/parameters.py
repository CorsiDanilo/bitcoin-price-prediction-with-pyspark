from imports import *

'''
Description: Returns the default parameters of the selected model
Args:
    model_name: Model name selected
Return: 
    params: Parameters of the selected model
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
            'regParam' : [0],
        }
    elif (model_name == 'RandomForestRegressor'):
        params = {
            'numTrees' : [20],
            'maxDepth' : [5]
            }
    elif (model_name == 'GBTRegressor'):
        params = {
            'maxIter' : [20],
            'maxDepth' : [5],
            'stepSize': [0.1]
        }
    
    return params

'''
Description: Returns the model grid parameters of the selected model
Args:
    model_name: Model name selected
Return: 
    params: Parameters of the selected model
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
            'maxDepth' : [3, 5, 10]
        }
    elif (model_name == 'GBTRegressor'):
        params = {
            'maxIter' : [5, 10, 20, 30, 40],
            'maxDepth' : [5, 8, 10],
            'stepSize': [0.1, 0.3, 0.5, 0.7]
        }

    return params

'''
Description: Returns the best model parameters of the selected model
Args:
    model_name: Model name selected
Return: 
    params: Parameters of the selected model
'''
def get_best_model_params(model_name):
    if (model_name == 'LinearRegression'):
        params = {
                'maxIter' : [5],
                'regParam' : [0.4],
                'elasticNetParam' : [0.2]
        }   
    if (model_name == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5],
            'regParam' : [0.2],
            'family': ['gaussian'],
            'link': ['log']
        }
    elif (model_name == 'RandomForestRegressor'):
        params = {
            'numTrees' : [30],
            'maxDepth' : [10]
            }
    elif (model_name == 'GBTRegressor'):
        params = {
            'maxIter' : [10],
            'maxDepth' : [5],
            'stepSize': [0.3]
        }
        
    return params