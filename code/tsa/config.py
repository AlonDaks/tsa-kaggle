import os
import sys
import re

# Python specific constants
path_to_file = os.path.realpath(__file__)

REPO_HOME_PATH = re.sub(r'/code/tsa/config\.py(c)*', '', path_to_file)

LARGE_DATA_BIN = os.environ['TSA_KAGGLE_DATA_DIR'] + '/'
