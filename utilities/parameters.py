from imports import *

def get_defaults_model_params(modelName):
    if (modelName == 'LinearRegression'):
        params = {
            'maxIter' : [100], # max number of iterations (>=0), default:100
            'regParam' : [0.0],# regularization parameter (>=0), default:0.0
            'elasticNetParam' : [0.0] # the ElasticNet mixing parameter, [0, 1], default:0.0
        }   
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [25], # max number of iterations (>=0), default:25
            'regParam' : [0], # regularization parameter (>=0), default:0.0
            'family': ['gaussian'], # The name of family which is a description of the error distribution to be used in the model.
            'link': ['identity'] # which provides the relationship between the linear predictor and the mean of the distribution function.
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [20],# Number of trees to train, >=1, default:20
            'maxDepth' : [5] # Maximum depth of the tree, <=30, default:5
            }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [20], # max number of iterations (>=0), default:20
            'maxDepth' : [5], # Maximum depth of the tree (>=0), <=30, default:5
            'stepSize': [0.1] # learning rate, [0,1], default:0.1
        }
    
    return params

def get_model_params(modelName):
    if (modelName == 'LinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80, 100], # max number of iterations (>=0), default:100
            'regParam' : np.arange(0,1,0.2).round(decimals=2),# regularization parameter (>=0), default:0.0
            'elasticNetParam' : np.arange(0,1,0.2).round(decimals=2) # the ElasticNet mixing parameter, [0, 1], default:0.0
        }
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5, 10, 50, 80], # max number of iterations (>=0), default:25
            'regParam' : [0.0, 0.1, 0.2], # regularization parameter (>=0), default:0.0
            # 'family': ['gaussian', 'gamma'], # The name of family which is a description of the error distribution to be used in the model.
            'family': ['gaussian'], # The name of family which is a description of the error distribution to be used in the model.
            # 'link': ['identity', 'inverse'] # which provides the relationship between the linear predictor and the mean of the distribution function.
            'link': ['log'] # which provides the relationship between the linear predictor and the mean of the distribution function.
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [5, 10, 15, 20, 25], # Number of trees to train, >=1, default:20
            'maxDepth' : [2, 3, 5, 7, 10] # Maximum depth of the tree, <=30, default:5
        }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [10, 20, 30], # max number of iterations (>=0), default:20
            'maxDepth' : [3, 5, 8], # Maximum depth of the tree (>=0), <=30, default:5
            'stepSize': [0.1, 0.3, 0.5, 0.7] # learning rate, [0,1], default:0.1
        }

    return params

def get_tuned_model_params(modelName):
    if (modelName == 'LinearRegression'):
        params = {
            'maxIter' : [5], # max number of iterations (>=0), default:100
            'regParam' : [0.8],# regularization parameter (>=0), default:0.0
            'elasticNetParam' : [0.8] # the ElasticNet mixing parameter, [0, 1], default:0.0
        }   
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5], # max number of iterations (>=0), default:25
            'regParam' : [0.0], # regularization parameter (>=0), default:0.0
            'family': ['gaussian'], # The name of family which is a description of the error distribution to be used in the model.
            'link': ['log'] # which provides the relationship between the linear predictor and the mean of the distribution function.
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [20],# Number of trees to train, >=1, default:20
            'maxDepth' : [10] # Maximum depth of the tree, <=30, default:5
            }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [10], # max number of iterations (>=0), default:20
            'maxDepth' : [5], # Maximum depth of the tree (>=0), <=30, default:5
            'stepSize': [0.5] # learning rate, [0,1], default:0.1
        }
    
    return params