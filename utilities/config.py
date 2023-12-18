################
# --- MISC --- #
################

# Models
LR = "LinearRegression"
GLR = "GeneralizedLinearRegression"
RF = "RandomForestRegressor"
GBTR = "GradientBoostingTreeRegressor"

#Spltting methods
BS = "block_splits"
WFS = "walk_forward_splits"
SS = "single_split"

# For reproducibility
RANDOM_SEED = 42

###################
# --- DATASET --- #
###################

# Datasets names
DATASET_NAME = "bitcoin_blockchain_data_15min"
DATASET_TRAIN_VALID_NAME = "bitcoin_blockchain_data_15min_train_valid"
DATASET_TEST_NAME = "bitcoin_blockchain_data_15min_test"

####################
# --- FEATURES --- #
####################

# Features labels
FEATURES_LABEL = "features"
TARGET_LABEL = "next-market-price"

# Features names
FEATURES_CORRELATION_LABEL = "features_correlation"
BASE_FEATURES_LABEL = "base_features"
BASE_AND_MOST_CORR_FEATURES_LABEL = "base_and_most_corr_features"
BASE_AND_LEAST_CORR_FEATURES_LABEL = "base_and_least_corr_features"

##################
# --- MODELS --- #
##################

# Model names
LR_MODEL_NAME = "LinearRegression"
GLR_MODEL_NAME = "GeneralizedLinearRegression"
RF_MODEL_NAME = "RandomForestRegressor"
GBTR_MODEL_NAME = "GradientBoostingTreeRegressor"

###################
# --- RESULTS --- #
###################

# Splits names
BLOCK_SPLITS_NAME = "block_splits"
WALK_FORWARD_SPLITS_NAME = "walk_forward_splits"
SHORT_TERM_SPLITS_NAME = "single_split"