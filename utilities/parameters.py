from imports import *

def get_defaults_model_params(modelName):
    if (modelName == 'LinearRegression'):
        params = {
                'maxIter' : [100],
                'regParam' : [0.0],
                'elasticNetParam' : [0.0]
        }   
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [25],
            'regParam' : [0],
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [20],
            'maxDepth' : [5]
            }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [20],
            'maxDepth' : [5],
            'stepSize': [0.1]
        }
    
    return params

def get_model_grid_params(modelName):
    if (modelName == 'LinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80, 100],
            'regParam' : np.arange(0,1,0.2).round(decimals=2),
            'elasticNetParam' : np.arange(0,1,0.2).round(decimals=2)
        }
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80],
            'regParam' : [0, 0.1, 0.2],
            'family': ['gaussian', 'gamma'],
            'link': ['log', 'identity', 'inverse']
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [3, 5, 10, 20, 30],
            'maxDepth' : [3, 5, 10]
        }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [5, 10, 20, 30, 40],
            'maxDepth' : [5, 8, 10],
            'stepSize': [0.1, 0.3, 0.5, 0.7]
        }

    return params