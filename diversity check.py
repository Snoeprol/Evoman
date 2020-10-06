import sys, os
os.chdir("C:/Users/igas/Evoman/")
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np
import copy
import random
import pandas as pd
import ast # read a data frame with lists
cwd = os.getcwd()
sys.argv = [1, '1', '1']
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

def read_data(file_path):
    def from_np_array(array_string):
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))
    return pd.read_csv(file_path, converters={'weights': from_np_array})


os.chdir("C:/Users/igas/Evoman/OutputData")
file_list = os.listdir()

df_variance = pd.DataFrame()
for file in file_list:
    location_gen = file.find("Generation")
    generation = file[location_gen+len("Generation")+1:location_gen+len("Generation")+3]
    if generation.find(",") == 1:
        generation = generation[0]
    location_runcode = file.find("Unique Runcode")
    run_code = file[location_runcode+len("Unique Runcode")+1:len(file)-4]
    data = read_data(file)
    data_columns = pd.DataFrame(data['weights'].to_list(), columns=range(len(data.weights[0])))
    variance = sum(data_columns.var(axis = 0))
    dict_df = {"run_code": [float(run_code)],
               "generation": [int(generation)],
                   "variance": [variance]}
    df = pd.DataFrame(dict_df)
    df_variance = df_variance.append(df)


df_variance.groupby(['run_code','generation'])['variance'].mean()

