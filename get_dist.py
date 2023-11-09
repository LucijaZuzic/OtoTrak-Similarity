import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
     
def process_time(time_as_str):
    time_as_str = time_as_str.split(".")[0]
    return datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S')

def point_from_traj_time_dist(one_point_time, one_traj_times):
    time_dist_arr = [abs(one_point_time - one_traj_time) for one_traj_time in one_traj_times]
    return min(time_dist_arr)

def point_from_traj_space_dist(one_point_long, one_point_lat, one_traj_longs, one_traj_lats):
    long_dist_arr = [abs(one_point_long - one_traj_long) for one_traj_long in one_traj_longs]
    lat_dist_arr = [abs(one_point_lat - one_traj_lat) for one_traj_lat in one_traj_lats]
    total_dist_arr = [np.sqrt(long_dist_arr[x] ** 2 + lat_dist_arr[x] ** 2) for x in range(len(one_traj_lats))]
    return min(total_dist_arr)
  
def points_from_traj_dist(timestamps1, longitudes1, latitudes1, timestamps2, longitudes2, latitudes2): 
    sum_time_d1 = 0

    for point_num in range(len(timestamps1)):
        time_d1 = point_from_traj_time_dist(timestamps1[point_num], timestamps2)
        sum_time_d1 += np.e ** (- time_d1)
        
    sum_time_d1 /= len(timestamps1)

    sum_space_d1 = 0

    for point_num in range(len(longitudes1)):
        space_d1 = point_from_traj_space_dist(longitudes1[point_num], latitudes1[point_num], longitudes2, latitudes2)
        sum_space_d1 += np.e ** (- space_d1)

    sum_space_d1 /= len(longitudes1)

    return sum_time_d1, sum_space_d1

def traj_dist(lambda_par, timestamps1, longitudes1, latitudes1, timestamps2, longitudes2, latitudes2):
    sum_time_d1, sum_space_d1 = points_from_traj_dist(timestamps1, longitudes1, latitudes1, timestamps2, longitudes2, latitudes2)
    sum_time_d2, sum_space_d2 = points_from_traj_dist(timestamps2, longitudes2, latitudes2, timestamps1, longitudes1, latitudes1)
    sum_time = sum_time_d1 + sum_time_d2
    sum_space = sum_space_d1 + sum_space_d2
    return lambda_par * sum_space + (1 - lambda_par) * sum_time

def save_plot_longitude_latitudes_for_ride(longitudes, latitudes, image_name):  
    plt.plot(longitudes, latitudes, color = 'k', linewidth = 10) 
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom = False)
    ax.yaxis.set_tick_params(labelleft = False)  
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.savefig(image_name, bbox_inches = 'tight') 
    plt.clf()

def random_colors(num_colors):
    colors_set = []
    for x in range(num_colors):
        string_color = "#"
        while string_color == "#" or string_color in colors_set:
            string_color = "#"
            set_letters = "0123456789ABCDEF"
            for y in range(6):
                string_color += set_letters[np.random.randint(0, 16)]
        colors_set.append(string_color)
    return colors_set
    
def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def preprocess_long_lat(long_list, lat_list):
    x_dir = long_list[0] < long_list[-1]
    y_dir = lat_list[0] < lat_list[-1]

    quadrant = 0

    if x_dir == True and y_dir == True:
        quadrant = 1
    if x_dir == False and y_dir == True:
        quadrant = 2
    if x_dir == False and y_dir == False:
        quadrant = 3
    if x_dir == True and y_dir == False:
        quadrant = 4
 
    long_list2 = [x - min(long_list) for x in long_list]
    lat_list2 = [y - min(lat_list) for y in lat_list]
    if x_dir == False: 
        long_list2 = [max(long_list2) - x for x in long_list2]
    if y_dir == False:
        lat_list2 = [max(lat_list2) - y for y in lat_list2]

    return long_list2, lat_list2, quadrant 
 
window_size = 20
step_size = window_size
#step_size = 1
max_trajs = 100
name_extension = "_window_" + str(window_size) + "_step_" + str(step_size) + "_segments_" + str(max_trajs)

all_subdirs = os.listdir() 

all_possible_rides = dict() 
if os.path.isfile("all_possible_rides"):
    all_possible_rides = load_object("all_possible_rides")

all_possible_trajs = dict() 
if os.path.isfile("all_possible_trajs"):
    all_possible_trajs = load_object("all_possible_trajs")

if window_size not in all_possible_trajs:
    all_possible_trajs[window_size] = dict()
 
total_possible_trajs = 0 
'''
for subdir_name in all_subdirs:

    trajs_in_dir = 0
    
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
    
    if subdir_name not in all_possible_trajs[window_size]:
        all_possible_trajs[window_size][subdir_name] = dict()

    if subdir_name not in all_possible_rides:
        all_possible_rides[subdir_name] = dict()
    
    all_rides_cleaned = os.listdir(subdir_name + "/cleaned_csv/")
    
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = set()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        
    for some_file in all_files:  
        if some_file in bad_rides_filenames:
            #print("Skipped ride", some_file)
            continue
        #print("Used ride", some_file)

        only_num_ride = some_file.replace(".csv", "").replace("events_", "")
        
        trajs_in_ride = 0

        if only_num_ride not in all_possible_trajs[window_size][subdir_name]:
            all_possible_trajs[window_size][subdir_name][only_num_ride] = dict()
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"])
        timestamps = [abs((process_time(one_traj_time) - process_time(file_with_ride["time"][0])).total_seconds()) for one_traj_time in file_with_ride["time"]]
         
        #longitudes_transform, latitudes_transform, quadrant = preprocess_long_lat(longitudes, latitudes) 

        #if only_num_ride not in all_possible_rides[subdir_name]:
            #all_possible_rides[subdir_name][only_num_ride] = {"time": timestamps, "long": longitudes_transform, "lat": latitudes_transform}
  
        for x in range(0, len(longitudes) - window_size + 1, step_size):
            longitudes_tmp = longitudes[x:x + window_size]
            latitudes_tmp = latitudes[x:x + window_size]
            timestamps_tmp = timestamps[x:x + window_size] 
            timestamps_tmp = [abs(one_traj_time - timestamps_tmp[0]) for one_traj_time in timestamps_tmp]

            set_longs = set()
            set_lats = set()
            for tmp_long in longitudes_tmp:
                set_longs.add(tmp_long)
            for tmp_lat in latitudes_tmp:
                set_lats.add(tmp_lat)
                
            if len(set_lats) == 1 and len(set_longs) == 1:
                continue  
            
            longitudes_tmp_transform, latitudes_tmp_transform, quadrant_tmp = preprocess_long_lat(longitudes_tmp, latitudes_tmp) 
            
            if np.max(longitudes_tmp_transform) > 0.5 or np.max(latitudes_tmp_transform) > 0.5:
                print(window_size, subdir_name, only_num_ride)
                plt.plot(longitudes_tmp_transform, latitudes_tmp_transform)
                plt.show()

            total_possible_trajs += 1
            trajs_in_ride += 1
            trajs_in_dir += 1 

            if x not in all_possible_trajs[window_size][subdir_name][only_num_ride]:
                all_possible_trajs[window_size][subdir_name][only_num_ride][x] = {"time": timestamps_tmp, "long": longitudes_tmp_transform, "lat": latitudes_tmp_transform}

        print(only_num_ride, trajs_in_ride)
    print(subdir_name, trajs_in_dir)
print(total_possible_trajs)
save_object("all_possible_trajs", all_possible_trajs)
'''
#save_object("all_possible_rides", all_possible_rides)
''''
for vehicle1 in all_possible_rides.keys():
    for vehicle2 in all_possible_rides.keys():
        if vehicle1 == vehicle2:
            continue
        print(vehicle1, vehicle2)
        distances_two_vehicles = []
        for r1 in all_possible_rides[vehicle1]:
            for r2 in all_possible_rides[vehicle2]: 
                print(r1, r2)
                distances_two_rides = [] 
                t1 = all_possible_rides[vehicle1][r1]
                t2 = all_possible_rides[vehicle2][r2] 
                plt.plot(t1["long"], t1["lat"])
                plt.show()
                plt.plot(t2["long"], t2["lat"])
                plt.show()
                one_d = traj_dist(0.3, t1["time"], t1["long"], t1["lat"], t2["time"], t2["long"], t2["lat"]) 
                distances_two_vehicles.append(one_d)  
        print(np.average(distances_two_vehicles), np.max(distances_two_vehicles), np.min(distances_two_vehicles), np.var(distances_two_vehicles), np.std(distances_two_vehicles))
        plt.hist(distances_two_vehicles)
        plt.show()
'''
for vehicle1 in all_possible_trajs[window_size].keys():
    for vehicle2 in all_possible_trajs[window_size].keys():
        if vehicle1 == vehicle2:
            continue
        print(vehicle1, vehicle2)
        distances_two_vehicles = []
        for r1 in all_possible_trajs[window_size][vehicle1]:
            for r2 in all_possible_trajs[window_size][vehicle2]: 
                print(r1, r2)
                distances_two_rides = [] 
                x12_pair = [] 
                for x1 in all_possible_trajs[window_size][vehicle1][r1]:
                    for x2 in all_possible_trajs[window_size][vehicle2][r2]:
                        t1 = all_possible_trajs[window_size][vehicle1][r1][x1]
                        t2 = all_possible_trajs[window_size][vehicle2][r2][x2] 
                        one_d = traj_dist(0.5, t1["time"], t1["long"], t1["lat"], t2["time"], t2["long"], t2["lat"])
                        distances_two_rides.append(one_d)  
                        distances_two_vehicles.append(one_d) 
                        x12_pair.append((x1, x2))
                print(np.average(distances_two_rides), np.max(distances_two_rides), np.min(distances_two_rides), np.var(distances_two_rides), np.std(distances_two_rides))
                plt.hist(distances_two_rides)
                plt.show()
                for x in range(len(distances_two_rides)):
                    if distances_two_rides[x] == np.max(distances_two_rides):
                        l1 = x12_pair[x][0]
                        l2 = x12_pair[x][1]
                        f1 = all_possible_trajs[window_size][vehicle1][r1][l1]
                        f2 = all_possible_trajs[window_size][vehicle2][r2][l2]
                        one_d = traj_dist(0.5, f1["time"], f1["long"], f1["lat"], f2["time"], f2["long"], f2["lat"])
                        print("Max", one_d, np.max(distances_two_rides))
                        plt.subplot(2, 2, 1)
                        plt.title("Max 1 " + str(l1))
                        plt.plot(f1["long"], f1["lat"])
                        plt.subplot(2, 2, 2)
                        plt.title("Max 2 " + str(l2))
                        plt.plot(f2["long"], f2["lat"])
                    if distances_two_rides[x] == np.min(distances_two_rides):
                        l1 = x12_pair[x][0]
                        l2 = x12_pair[x][1]
                        f1 = all_possible_trajs[window_size][vehicle1][r1][l1]
                        f2 = all_possible_trajs[window_size][vehicle2][r2][l2]
                        one_d = traj_dist(0.5, f1["time"], f1["long"], f1["lat"], f2["time"], f2["long"], f2["lat"])
                        print("Min", one_d, np.min(distances_two_rides))
                        plt.subplot(2, 2, 3)
                        plt.title("Min 1 " + str(l1))
                        plt.plot(f1["long"], f1["lat"])
                        plt.subplot(2, 2, 4)
                        plt.title("Min 2 " + str(l2))
                        plt.plot(f2["long"], f2["lat"])
                plt.plot()
                plt.show()
                break
        print(np.average(distances_two_vehicles), np.max(distances_two_vehicles), np.min(distances_two_vehicles), np.var(distances_two_vehicles), np.std(distances_two_vehicles))
        plt.hist(distances_two_vehicles)
        plt.show()