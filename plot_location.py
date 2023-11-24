import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle
import numpy as np
    
def process_time(time_as_str):
    time_as_str = time_as_str.split(".")[0]
    return (datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds() + milisecond / 1000

def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
all_subdirs = os.listdir() 
'''
for subdir_name in all_subdirs:
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    print(subdir_name)
      
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = dict()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
    maxx = -100000
    minx = 100000
    maxy = -100000
    miny = 100000
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames:
            #print("Skipped ride", some_file)
            continue
        #print("Used ride", some_file)
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"]) 
        maxx = max(max(longitudes), maxx)
        minx = min(min(longitudes), minx)
        maxy = max(max(latitudes), maxy)
        miny = min(min(latitudes), miny)
        plt.plot(longitudes, latitudes)
        
    plt.title(subdir_name)
    plt.xticks([minx, maxx], [np.round(minx, 3), np.round(maxx, 3)]) 
    plt.yticks([miny, maxy], [np.round(miny, 3), np.round(maxy, 3)]) 
    #plt.savefig(subdir_name + "/" + subdir_name + "_trajectories.png", bbox_inches = "tight")
    plt.show()
'''
for subdir_name in ["Vehicle_1", "Vehicle_10"]:
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    print(subdir_name)
      
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = dict()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames:
            #print("Skipped ride", some_file)
            continue
        #print("Used ride", some_file)
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"])  
        plt.plot(longitudes, latitudes) 
        plt.title(subdir_name + " " + some_file.replace(".csv", ""))
        plt.xticks([min(longitudes), max(longitudes)], [np.round(min(longitudes), 3), np.round(max(longitudes), 3)]) 
        plt.yticks([min(latitudes), max(latitudes)], [np.round(min(latitudes), 3), np.round(max(latitudes), 3)])  
        plt.show()