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
            'maxIter' : [5, 50, 100, 120, 150], # max number of iterations (>=0), default:100
            'regParam' : [0.0, 0.05, 0.1], # regularization parameter (>=0), default:0.0
            'elasticNetParam' : [0.0, 0.5, 1.0] # the ElasticNet mixing parameter, [0, 1], default:0.0
        }
    if (modelName == 'GeneralizedLinearRegression'):
        params = {
            'maxIter' : [5, 15, 25, 50, 80], # max number of iterations (>=0), default:25
            'regParam' : [0.0, 0.05, 0.1], # regularization parameter (>=0), default:0.0
            'family': ['gaussian'], # The name of family which is a description of the error distribution to be used in the model.
            'link': ['log'] # which provides the relationship between the linear predictor and the mean of the distribution function.
        }
    elif (modelName == 'RandomForestRegressor'):
        params = {
            'numTrees' : [10, 20, 30], # Number of trees to train, >=1, default:20
            'maxDepth' : [3, 5, 8] # Maximum depth of the tree, <=30, default:5
        }
    elif (modelName == 'GBTRegressor'):
        params = {
            'maxIter' : [10, 20, 30], # max number of iterations (>=0), default:20
            'maxDepth' : [3, 5, 8], # Maximum depth of the tree (>=0), <=30, default:5
            'stepSize': [0.1, 0.5, 0.7] # learning rate, [0,1], default:0.1
        }

    return params