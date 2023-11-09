import pandas as pd
from datetime import datetime
import os
import pickle
     
def process_time(time_as_str):
    time_as_str = time_as_str.split(".")[0]
    return datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S')

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

for subdir_name in all_subdirs:
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    print(subdir_name)

    all_files = os.listdir(subdir_name + "/csv_for_rides/")
    all_set = set()
    for one_file in all_files:
        all_set.add(subdir_name + "/csv_for_rides/" + one_file)

    found_tour = False
    for file_some in all_files:
        if "tours" in file_some:
            found_tour = True
            file_with_tours = pd.read_csv(subdir_name + "/csv_for_rides/" + file_some) 
            break
    if not found_tour:
        print(subdir_name, "no tour")
        continue

    bad_rides_filenames = dict()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")

    max_start_found = process_time("2023-08-01 00:00:00.000")
    min_start_found = process_time("2023-07-01 00:00:00.000") 
    stuff_to_print = ""
    total_len = set()
    in_time = set()
    no_end = set()
    errored = set()
    tracking = set()
    processed = set()
    not_processed = set()
    no_pair = set()
    for entry_num in range(len(file_with_tours["asset_id"])):
        name_new_file = subdir_name + "/cleaned_csv/events_" + str(file_with_tours["id"][entry_num]) + ".csv"
        total_len.add(name_new_file)
        datetime_start = process_time(file_with_tours["start"][entry_num])
        if str(file_with_tours["end"][entry_num]) == 'nan':
            no_end.add(name_new_file)
            bad_rides_filenames[name_new_file] = -3
            continue
        datetime_end = process_time(file_with_tours["end"][entry_num])
        error_mode = file_with_tours["operation_mode_name"][entry_num]
        if datetime_start < min_start_found or datetime_end >= max_start_found: 
            bad_rides_filenames[name_new_file] = -4
            continue  
        in_time.add(name_new_file)
        if error_mode == "ERROR":
            #print("ERROR") 
            errored.add(name_new_file)
            bad_rides_filenames[name_new_file] = -1
            continue
        if error_mode == "TRACKING":
            #print("TRACKING") 
            tracking.add(name_new_file)
            bad_rides_filenames[name_new_file] = -2
            continue
        if os.path.isfile(name_new_file):
            #print("CLEAN") 
            processed.add(name_new_file)
            continue
        not_processed.add(name_new_file)
        stuff_to_print += "SELECT * FROM events WHERE asset_id = " + str(file_with_tours["asset_id"][entry_num]) + " AND time >= timestamp '" + file_with_tours["start"][entry_num] + "' AND time <= timestamp '" + file_with_tours["end"][entry_num] + "';\n"
        #print("SELECT * FROM events WHERE asset_id = " + str(file_with_tours["asset_id"][entry_num]) + " AND time >= timestamp '" + file_with_tours["start"][entry_num] + "' AND time <= timestamp '" + file_with_tours["end"][entry_num] + "';")
    
    if not os.path.isdir(subdir_name + "/cleaned_csv/"):
        os.makedirs(subdir_name + "/cleaned_csv/")
    all_clean_files = os.listdir(subdir_name + "/cleaned_csv/")
    all_clean_set = set()
    for clean_file in all_clean_files:
        all_clean_set.add(subdir_name + "/cleaned_csv/" + clean_file)

    if len(all_clean_set.difference(processed)) > 0 or len(all_set) != len(all_clean_set) + 1 or len(not_processed) > 0:
        print("total", len(total_len), "no end", len(no_end))
        print("in time", len(in_time), "errored", len(errored), "tracking", len(tracking), "processed", len(processed), "not processed", len(not_processed))
        print("all clean", len(all_clean_set), "all clean and processed", len(all_clean_set.intersection(processed)), "extra", len(all_clean_set.difference(processed)))
        print("all set", len(all_set))
 
    if stuff_to_print != "":
        file_to_print = open(subdir_name + "/SELECT_query.sql", "w")
        file_to_print.write(stuff_to_print)
        file_to_print.close()
        print("Saved")
    else:
        if os.path.isfile(subdir_name + "/SELECT_query.sql"):
            os.remove(subdir_name + "/SELECT_query.sql")

    save_object(subdir_name + "/bad_rides_filenames", bad_rides_filenames)
