#Chu, Chen
#For orginal cityflow
import cityflow
import datetime
import json
import numpy as np
from tqdm import tqdm


# this is the speed tracking, on simple setting: (no lane changing, no traffic light control)
# full vehicle appearance!

# simple test case:
path = "/home/realCityFlow/realCityFlow/experiment/cityflow8x8/config.json"

# atlanta test case:
# path = "/home/realCityFlow/realCityFlow/experiment/atlanta_data/config.json"

eng = cityflow.Engine(path, thread_num=8)
print("staring successfully....")


time_list  = []

for i in tqdm(range(1)):

    time_now = datetime.datetime.now() 
    for step in range(200):

        eng.next_step()

    time_var = (datetime.datetime.now() - time_now).total_seconds()
    time_list.append(time_var)

print(time_list)
print(np.array(time_list).mean())
print(np.array(time_list).std())

