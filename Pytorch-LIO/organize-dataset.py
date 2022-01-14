import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# Parameters
orig_folder = "Kvasir v2"
targ_folder_train = "kvasirV2/train"
targ_folder_valid = "kvasirV2/valid"
targ_folder_test = "kvasirV2/test"

for _dir in [targ_folder_train, targ_folder_valid, targ_folder_test]:
    if not os.path.exists(_dir):
        os.makedirs(_dir)

classes = os.listdir(orig_folder)
# store class indices represented by each class
lIdx, classDict = 0, dict()
# store the path, class and class index of each image
orig_path, targ_path, label, indices = [], [], [], [] 
for clss in classes:
    orig_dir = "{}/{}/{}".format(orig_folder, clss, clss)
    # produce corresponding index for the class
    if not clss in classDict.keys():
        lIdx += 1
        classDict[clss] = lIdx
    # for each file, store them into the same folder
    files = os.listdir(orig_dir)
    for file in files:
        orig_path.append('{}/{}'.format(orig_dir, file))
        label.append(clss)
        indices.append(classDict[clss])
        

# save class-index dictionary into json file
with open('class-index.json', 'w') as fp:
    json.dump(classDict, fp)

del(classDict)
del(lIdx)



# Stat all lists into a dataframe
df = pd.DataFrame(data={"orig_path": orig_path, "label": label, "index": indices})
print("============ Dataset Overall ======================")
print(df.shape)
print(df.head(5))

# splitting dataset into training, validation, testing dataset
df_train, df_test = train_test_split(df, test_size=0.15, random_state=256, shuffle=True)
df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=256, shuffle=True)
print("Train: {}\nValid: {}\nTest: {}".format(df_train.shape, df_valid.shape, df_test.shape))
del(df)

# store dataset
def organized(df, fold):
    new_path = []
    for idx, orig_path in enumerate(df["orig_path"].values):
        targ_path = '{}/{:05d}.jpg'.format(fold, idx + 1)
        new_path.append('{:05d}.jpg'.format(idx + 1))
        # save image into the same target folder
        shutil.copy(orig_path, targ_path)

    # renew dataset columns
    df.drop("orig_path", axis=1)
    df["ImagePath"] = new_path

organized(df_train, targ_folder_train)
organized(df_valid, targ_folder_valid)
organized(df_test, targ_folder_test)

# save dataframe
df_train.to_csv("training.csv", index=False)
df_valid.to_csv("validation.csv", index=False)
df_test.to_csv("testing.csv", index=False)