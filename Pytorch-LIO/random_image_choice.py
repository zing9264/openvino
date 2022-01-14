import os
import random
import shutil
import argparse
import warnings
import pandas as pd

# prevent showing warnings
warnings.filterwarnings("ignore")

# python random_image_choice.py  --image_num 10
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--orig_dir', default='kvasirV2/test/', type=str)
parser.add_argument('--target_dir', default='kvasirV2/demo/', type=str)
parser.add_argument('--image_num', default=10, type=int)
parser.add_argument('--df_dir', default='demo.csv', type=str)
args = parser.parse_args()

# Params
ORIG_DIR = args.orig_dir
TARG_DIR = args.target_dir
IMG_NUM = args.image_num
DF_DIR = args.df_dir
try:
    shutil.rmtree(TARG_DIR)
except:
    pass
if not os.path.exists(TARG_DIR):
    os.makedirs(TARG_DIR)

df_test = pd.read_csv(DF_DIR, dtype={'ImagePath': str, 'index': int})
files = random.sample(df_test['ImagePath'].values.tolist(), k=IMG_NUM)
i=0
for file in files:
    i+=1
    shutil.copy(ORIG_DIR + file, TARG_DIR + file)

