import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
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
    for file_some in all_files:
        if "tours" in file_some:
            file_with_tours = pd.read_csv(subdir_name + "/csv_for_rides/" + file_some) 
           
    max_start_found = process_time("2012-07-01 00:00:00.000")
    min_start_found = process_time("2032-07-01 00:00:00.000")  
 
    asc_rides = set()
    desc_rides = set()  
    mixed_rides = set() 
 
    ride_for_file = dict() 

    files_seen = set() 
    bad_rides_filenames = dict()
    good_rides_filenames = dict()
    
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")

    if not os.path.isdir(subdir_name + "/cleaned_csv/"):
        os.makedirs(subdir_name + "/cleaned_csv/")

    max_gap = process_time("2012-07-01 00:00:00.000") - process_time("2012-07-01 00:00:00.000")

    sorted_set = set()
    reverse_set = set()
    unsorted_set = set()       
           
    for some_file in all_files: 
        if some_file[0] == "e" and ".csv" in some_file: 
            file_with_ride = pd.read_csv(subdir_name + "/csv_for_rides/" + some_file) 
            list_time = list(file_with_ride["time"])
 
            sgn_set = set()
            for x in range(len(list_time)):
                list_time[x] = process_time(list_time[x])
                if x > 0:
                    sgn_set.add(list_time[x] > list_time[x - 1]) 
         
            found_match = 0
            id_entry = -1
            
            for entry_num in range(len(file_with_tours["asset_id"])):
                datetime_start = process_time(file_with_tours["start"][entry_num])
                datetime_end = process_time(file_with_tours["end"][entry_num])
                if min(list_time) >= datetime_start and max(list_time) <= datetime_end:
                    #print("yes", file_with_tours["id"][entry_num], datetime_start, datetime_end)
                    predicted = (datetime_end - datetime_start).total_seconds()
                    average_time = predicted / len(list_time)
                    #print("measurements", len(list_time))
                    #print("predicted time", predicted, datetime_end - datetime_start) 
                    #print("average time", average_time) 
                    id_entry = file_with_tours["id"][entry_num]
                    found_match += 1  
                
            file_name_new = subdir_name + "/cleaned_csv/events_" + str(id_entry) + ".csv"
            if found_match > 1:  
                print("Unclear", some_file) 
            if list(ride_for_file.values()).count(file_name_new) > 0:  
                print("Duplicate found", some_file)  
                os.remove(subdir_name + "/csv_for_rides/" + some_file)
                continue
                        
            if found_match == 0:    
                print(some_file, " no match") 
                continue
                
            if len(sgn_set) == 1 and False in sgn_set:
                reverse_set.add(some_file)
            if len(sgn_set) == 1 and True in sgn_set:
                sorted_set.add(some_file) 
            if len(sgn_set) > 1:
                unsorted_set.add(some_file)
            #print(sgn_set)
        
            if some_file in reverse_set:
                list_time_sorted = [list_time[len(list_time) - 1 - x] for x in range(len(list_time))]
            if some_file in sorted_set:
                list_time_sorted = [ele for ele in list_time]
            if some_file in unsorted_set:
                list_time_sorted = sorted(list_time)
            
            max_gap_for_ride = process_time("2012-07-01 00:00:00.000") - process_time("2012-07-01 00:00:00.000")
            for x in range(len(list_time_sorted) - 1):
                gap_for_ride = list_time_sorted[x + 1] - list_time_sorted[x]
                max_gap_for_ride = max(max_gap_for_ride, gap_for_ride)
                max_gap = max(max_gap, max_gap_for_ride)
            #print("max gap for ride", max_gap_for_ride) 
                
            if max_gap_for_ride.total_seconds() < 5:
                if file_name_new not in bad_rides_filenames or (file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0):
                    good_rides_filenames[file_name_new] = max_gap_for_ride.total_seconds()
                if file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0: 
                    bad_rides_filenames.pop(file_name_new)
            else:
                if file_name_new not in bad_rides_filenames or (file_name_new in bad_rides_filenames and bad_rides_filenames[file_name_new] > 0):
                    bad_rides_filenames[file_name_new] = max_gap_for_ride.total_seconds()  
            ride_for_file[some_file] = file_name_new

            if os.path.isfile(file_name_new):
                #print("Already processed", some_file)   
                continue

            cols_of_csv = list(file_with_ride.columns)
            for y in range(len(file_with_ride["fields"])):
                dictionary_fields = eval(file_with_ride["fields"][y].replace("true", "True").replace("false", "False"))  
                for key_field in dictionary_fields: 
                    if "fields_" + key_field not in cols_of_csv:
                        cols_of_csv.append("fields_" + key_field) 

            write_csv_content = ""
            for col_name in cols_of_csv:
                if col_name == "fields":
                    continue
                if write_csv_content != "":
                    write_csv_content += ","
                write_csv_content += '"' + col_name + '"' 
            write_csv_content += "\n" 

            original_indexes = [x for x in range(len(list_time))]

            if some_file in reverse_set:
                new_indexes = [len(list_time) - 1 - x for x in range(len(list_time))] 
            if some_file in sorted_set:
                new_indexes = [x for x in range(len(list_time))] 
            if some_file in unsorted_set:
                new_indexes = [list_time.index(list_time_sorted[x]) for x in range(len(list_time))] 

            new_indexes_reversed = [new_indexes[x] for x in range(len(new_indexes))]
            new_indexes.reverse() 

            for row_num in new_indexes_reversed:
                col_index = 0
                for col_name in cols_of_csv: 
                    if "fields" not in col_name:
                        if col_index != 0:
                            write_csv_content += ","
                        write_csv_content += '"' + str(file_with_ride[col_name][row_num]) + '"' 
                        col_index += 1
                dictionary_fields = eval(file_with_ride["fields"][row_num].replace("true", "True").replace("false", "False")) 
                for col_name in cols_of_csv:
                    if "fields_" in col_name:
                        if col_name.replace("fields_", "") in dictionary_fields:
                            write_csv_content += ',"' + str(dictionary_fields[col_name.replace("fields_", "")]) + '"' 
                        else: 
                            write_csv_content += ',""' 
                write_csv_content += "\n" 
            #print(write_csv_content)  
                
            file_new_csv = open(file_name_new, "w")
            file_new_csv.write(write_csv_content)
            file_new_csv.close() 
    
    print(len(sorted_set), len(reverse_set), len(unsorted_set)) 
    print(len(bad_rides_filenames), len(good_rides_filenames), len(ride_for_file)) 

    save_object(subdir_name + "/bad_rides_filenames", bad_rides_filenames)
