import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np 
import pickle
      
def get_vector(x1, x2, y1, y2):
    if x1 == x2:
        return 0, -1, x1 
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1 
        return 1, a, b 
        
def get_intersection(x1, x2, y1, y2, x3, x4, y3, y4):
    yc1, a1, b1 = get_vector(x1, x2, y1, y2)
    yc2, a2, b2 = get_vector(x3, x4, y3, y4)
    if yc1 != 0 and yc2 != 0:
        if a1 != a2:
            xs = (b2 - b1) / (a1 - a2)
            ys = a1 * xs + b1
            return xs, ys
        else:
            return "Nan", "Nan"
    else:
        if yc1 == 0 and yc2 != 0:
            return x1, a2 * x1 + b2
        if yc2 == 0 and yc1 != 0:
            return x3, a1 * x3 + b1
        return "Nan", "Nan"
        
def point_on_line(xs, ys, x1, x2, y1, y2): 
    minx = min(x1, x2)
    maxx = max(x1, x2)
    miny = min(y1, y2)
    maxy = max(y1, y2) 
    return xs >= minx and xs <= maxx and ys >= miny and ys <= maxy
    
def get_surface(x1, x2, x3, y1, y2, y3):
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y2 - y1))
    
def traj_segment_dist(x1, y1, x2, y2, x3, y3, x4, y4):
    xs, ys = get_intersection(x1, x2, y1, y2, x3, x4, y3, y4)
    s1 = get_surface(x2, x3, x4, y2, y3, y4)
    s2 = get_surface(x1, x3, x4, y1, y3, y4)
    s3 = get_surface(x1, x2, x4, y1, y2, y4)
    s4 = get_surface(x1, x2, x3, y1, y2, y3)
    if xs == "Nan":
        return (s1 + s2 + s3 + s4) / 2
    else:
        if point_on_line(xs, ys, x3, x4, y3, y4) or point_on_line(xs, ys, x1, x2, y1, y2):
            s5 = get_surface(x1, x2, xs, y1, y2, ys)
            s6 = get_surface(x3, x4, xs, y3, y4, ys)
            return s5 + s6
        else:
            return (s1 + s2 + s3 + s4) / 2
        
def traj_segment_dist_markings(color_paint, x1, y1, x2, y2, x3, y3, x4, y4):
    xs, ys = get_intersection(x1, x2, y1, y2, x3, x4, y3, y4)
    s1 = get_surface(x2, x3, x4, y2, y3, y4)
    s2 = get_surface(x1, x3, x4, y1, y3, y4)
    s3 = get_surface(x1, x2, x4, y1, y2, y4)
    s4 = get_surface(x1, x2, x3, y1, y2, y3)
    plt.plot([x1, x2], [y1, y2], color_paint)
    plt.plot([x1, x3], [y1, y3], color_paint)
    plt.plot([x1, x4], [y1, y4], color_paint)
    plt.plot([x2, x3], [y2, y3], color_paint)
    plt.plot([x2, x4], [y2, y4], color_paint)
    plt.plot([x3, x4], [y3, y4], color_paint)
    if xs == "Nan":
        return (s1 + s2 + s3 + s4) / 2
    else:
        if point_on_line(xs, ys, x3, x4, y3, y4) or point_on_line(xs, ys, x1, x2, y1, y2):
            s5 = get_surface(x1, x2, xs, y1, y2, ys)
            s6 = get_surface(x3, x4, xs, y3, y4, ys)
            return s5 + s6
        else:
            return (s1 + s2 + s3 + s4) / 2
     
def traj_dist(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1) - 1):
        sum_dist += traj_segment_dist(longitudes1[i], latitudes1[i], longitudes1[i + 1], latitudes1[i + 1], longitudes2[i], latitudes2[i], longitudes2[i + 1], latitudes2[i + 1])
    return sum_dist

def traj_dist_markings(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    colors_set = random_colors(len(longitudes1) - 1)
    for i in range(len(longitudes1) - 1):
        sum_dist += traj_segment_dist_markings(colors_set[i], longitudes1[i], latitudes1[i], longitudes1[i + 1], latitudes1[i + 1], longitudes2[i], latitudes2[i], longitudes2[i + 1], latitudes2[i + 1])
    return sum_dist
    
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
 
    long_list2 = [x - min(long_list) for x in long_list]
    lat_list2 = [y - min(lat_list) for y in lat_list]
    if x_dir == False: 
        long_list2 = [max(long_list2) - x for x in long_list2]
    if y_dir == False:
        lat_list2 = [max(lat_list2) - y for y in lat_list2]

    return long_list2, lat_list2    
    
def scale_long_lat_min_max(long_list, lat_list, minx, miny, maxx, maxy):
    x_diff = maxx - minx
    if x_diff == 0:
        x_diff = 1
    y_diff = maxy - miny 
    if y_diff == 0:
        y_diff = 1
    long_list2 = [(x - min(long_list)) / x_diff for x in long_list]
    lat_list2 = [(y - min(lat_list)) / y_diff for y in lat_list]
    return long_list2, lat_list2 
    
def scale_long_lat(long_list, lat_list):
    minx = np.min(long_list)
    maxx = np.max(long_list)
    miny = np.min(lat_list)
    maxy = np.max(lat_list)
    return scale_long_lat_min_max(long_list, lat_list, minx, miny, maxx, maxy)
 
window_size = 20
step_size = window_size
#step_size = 1
max_trajs = 100
name_extension = "_window_" + str(window_size) + "_step_" + str(step_size) + "_segments_" + str(max_trajs)

all_subdirs = os.listdir() 
  
all_possible_trajs = dict()   
all_possible_trajs[window_size] = dict()

trajectory_monotonous = dict()
trajectory_monotonous[window_size] = dict()
label_NF = 0
label_NM = 0
label_I = 0
label_D = 0
 
total_possible_trajs = 0

for subdir_name in all_subdirs:

    trajs_in_dir = 0
    
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
     
    all_possible_trajs[window_size][subdir_name] = dict() 
    trajectory_monotonous[window_size][subdir_name] = dict() 

    all_rides_cleaned = os.listdir(subdir_name + "/cleaned_csv/")
    
    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = set()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames:
            #print("Skipped ride", some_file)
            continue
        #print("Used ride", some_file)

        only_num_ride = some_file.replace(".csv", "").replace("events_", "")
        
        trajs_in_ride = 0

        all_possible_trajs[window_size][subdir_name][only_num_ride] = dict()
        trajectory_monotonous[window_size][subdir_name][only_num_ride] = dict()
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"]) 
  
        for x in range(0, len(longitudes) - window_size + 1, step_size):
            longitudes_tmp = longitudes[x:x + window_size]
            latitudes_tmp = latitudes[x:x + window_size]

            set_longs = set()
            set_lats = set()
            for tmp_long in longitudes_tmp:
                set_longs.add(tmp_long)
            for tmp_lat in latitudes_tmp:
                set_lats.add(tmp_lat)
                
            if len(set_lats) == 1 and len(set_longs) == 1:
                continue  
            
            longitudes_tmp_transform, latitudes_tmp_transform = preprocess_long_lat(longitudes_tmp, latitudes_tmp)
            
            longitudes_tmp_transform, latitudes_tmp_transform = scale_long_lat(longitudes_tmp_transform, latitudes_tmp_transform)
              
            total_possible_trajs += 1
            trajs_in_ride += 1
            trajs_in_dir += 1 

            all_possible_trajs[window_size][subdir_name][only_num_ride][x] = {"long": longitudes_tmp_transform, "lat": latitudes_tmp_transform}

            long_sgn = set()
            for long_ind in range(len(longitudes_tmp_transform) - 1):
                long_sgn.add(longitudes_tmp_transform[long_ind + 1] > longitudes_tmp_transform[long_ind])
                if len(long_sgn) > 1:
                    break
                
            lat_sgn = set()
            for lat_ind in range(len(latitudes_tmp_transform) - 1):
                lat_sgn.add(latitudes_tmp_transform[lat_ind + 1] > latitudes_tmp_transform[lat_ind])
                if len(lat_sgn) > 1:
                    break
            
            if len(lat_sgn) > 1 and len(long_sgn) > 1:
                trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "NF"
                label_NF += 1
            if (len(lat_sgn) == 1 and len(long_sgn) > 1) or (len(lat_sgn) > 1 and len(long_sgn) == 1):
                trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "NM"
                label_NM += 1
            if len(lat_sgn) == 1 and len(long_sgn) == 1:
                if (True in lat_sgn and True in long_sgn) or (False in lat_sgn and False in long_sgn):
                    trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "I"
                    label_D += 1
                else:
                    trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "D"
                    label_I += 1
                 
        #print(only_num_ride, trajs_in_ride)
    print(subdir_name, trajs_in_dir)
print(total_possible_trajs)
print(label_NF, label_NM, label_D, label_I) 

def compare_all_with_sample(sample_x, sample_y, title_sample):
 
    distances_from_sample_ride_vehicle = [] 
    labels_from_sample_ride_vehicle = [] 

    max_distances_from_sample_ride_vehicle = 0
    index_max_distances_from_sample_ride_vehicle = 0 

    min_distances_from_sample_ride_vehicle = 100000
    index_min_distances_from_sample_ride_vehicle = 0 

    for vehicle1 in all_possible_trajs[window_size].keys():  
        for r1 in all_possible_trajs[window_size][vehicle1]:  
            for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
                t1 = all_possible_trajs[window_size][vehicle1][r1][x1]  

                td_up = traj_dist(t1["long"], t1["lat"], sample_x, sample_y) 
                distances_from_sample_ride_vehicle.append(td_up) 
                labels_from_sample_ride_vehicle.append(trajectory_monotonous[window_size][vehicle1][r1][x1])

                if distances_from_sample_ride_vehicle[-1] > max_distances_from_sample_ride_vehicle:
                    index_max_distances_from_sample_ride_vehicle = t1
                    max_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]
  
                if distances_from_sample_ride_vehicle[-1] < min_distances_from_sample_ride_vehicle:
                    index_min_distances_from_sample_ride_vehicle = t1
                    min_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]
    
    plt.subplot(1, 2, 1)
    plt.title("Min " + title_sample)
    plt.plot(index_min_distances_from_sample_ride_vehicle["long"], index_min_distances_from_sample_ride_vehicle["lat"])
    plt.plot(sample_x, sample_y)
    plt.subplot(1, 2, 2)
    plt.title("Max " + title_sample)
    plt.plot(index_max_distances_from_sample_ride_vehicle["long"], index_max_distances_from_sample_ride_vehicle["lat"])
    plt.plot(sample_x, sample_y)
    plt.show()

    only_NF = []
    only_NM = []
    only_I = []
    only_D = []
    for i in range(len(distances_from_sample_ride_vehicle)):
        if labels_from_sample_ride_vehicle[i] == "NF":
            only_NF.append(distances_from_sample_ride_vehicle[i])
        if labels_from_sample_ride_vehicle[i] == "NM":
            only_NM.append(distances_from_sample_ride_vehicle[i])
        if labels_from_sample_ride_vehicle[i] == "I":
            only_I.append(distances_from_sample_ride_vehicle[i])
        if labels_from_sample_ride_vehicle[i] == "D":
            only_D.append(distances_from_sample_ride_vehicle[i])
  
    plt.title("Dist " + title_sample)
    plt.hist(distances_from_sample_ride_vehicle)
    plt.hist(only_NF, label = "NF")
    plt.hist(only_NM, label = "NM")
    plt.hist(only_I, label = "I") 
    plt.legend()
    plt.show()

    return distances_from_sample_ride_vehicle, labels_from_sample_ride_vehicle, only_NF, only_NM, only_I

left_edge_x = [0 for i in range(window_size)]
left_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
left_dist, traj_labels, left_NF, left_NM, left_I = compare_all_with_sample(left_edge_x, left_edge_y, "left")

right_edge_x = [1 for i in range(window_size)]
right_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
right_dist, traj_labels, right_NF, right_NM, right_I = compare_all_with_sample(right_edge_x, right_edge_y, "right")

down_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
down_edge_y = [0 for i in range(window_size)]
down_dist, traj_labels, down_NF, down_NM, down_I = compare_all_with_sample(down_edge_x, down_edge_y, "down")

up_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
up_edge_y = [1 for i in range(window_size)]
up_dist, traj_labels, up_NF, up_NM, up_I = compare_all_with_sample(up_edge_x, up_edge_y, "up")

diagonal_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
diagonal_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
diagonal_edge_dist, traj_labels, diagonal_NF, diagonal_NM, diagonal_I = compare_all_with_sample(diagonal_edge_x, diagonal_edge_y, "diagonal")

left_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
left_circle_x = [np.sqrt(- y * (y - 1)) for y in left_circle_y]
left_circle_dist, traj_labels, left_circle_NF, left_circle_NM, left_circle_I = compare_all_with_sample(left_circle_x, left_circle_y, "left circle")

right_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
right_circle_x = [1 - np.sqrt(- y * (y - 1)) for y in right_circle_y]
right_circle_dist, traj_labels, right_circle_NF, right_circle_NM, right_circle_I = compare_all_with_sample(right_circle_x, right_circle_y, "right circle")

down_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
down_circle_y = [np.sqrt(- x * (x - 1)) for x in down_circle_x]
down_circle_dist, traj_labels, down_circle_NF, down_circle_NM, down_circle_I = compare_all_with_sample(down_circle_x, down_circle_y, "down circle") 

up_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
up_circle_y = [1 - np.sqrt(- x * (x - 1)) for x in up_circle_x] 
up_circle_dist, traj_labels, up_circle_NF, up_circle_NM, up_circle_I = compare_all_with_sample(up_circle_x, up_circle_y, "up circle") 
  
sin_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_y = [np.sin(x * np.pi * 2) for x in sin_x] 
sin_dist, traj_labels, sin_NF, sin_NM, sin_I = compare_all_with_sample(sin_x, sin_y, "sin")  

sin_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_reverse_y = [np.sin(x * np.pi * 2 + np.pi) for x in sin_x] 
sin_reverse_dist, traj_labels, sin_reverse_NF, sin_reverse_NM, sin_reverse_I = compare_all_with_sample(sin_reverse_x, sin_reverse_y, "sin reverse")   

sin_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_half_y = [np.sin(x * np.pi) for x in sin_half_x] 
sin_half_dist, traj_labels, sin_half_NF, sin_half_NM, sin_half_I = compare_all_with_sample(sin_half_x, sin_half_y, "sin half")   

sin_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_half_reverse_y = [np.sin(x * np.pi + np.pi) for x in sin_half_x]  
sin_half_reverse_dist, traj_labels, sin_half_reverse_NF, sin_half_reverse_NM, sin_half_reverse_I = compare_all_with_sample(sin_half_reverse_x, sin_half_reverse_y, "sin half reverse")   
 
cos_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_y = [np.cos(x * np.pi * 2) for x in cos_x] 
cos_dist, traj_labels, cos_NF, cos_NM, cos_I = compare_all_with_sample(cos_x, cos_y, "cos")  

cos_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_reverse_y = [np.cos(x * np.pi * 2 + np.pi) for x in cos_x] 
cos_reverse_dist, traj_labels, cos_reverse_NF, cos_reverse_NM, cos_reverse_I = compare_all_with_sample(cos_reverse_x, cos_reverse_y, "cos reverse")   

cos_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_half_y = [np.cos(x * np.pi) for x in cos_half_x] 
cos_half_dist, traj_labels, cos_half_NF, cos_half_NM, cos_half_I = compare_all_with_sample(cos_half_x, cos_half_y, "cos half")   

cos_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_half_reverse_y = [np.cos(x * np.pi + np.pi) for x in cos_half_x]  
cos_half_reverse_dist, traj_labels, cos_half_reverse_NF, cos_half_reverse_NM, cos_half_reverse_I = compare_all_with_sample(cos_half_reverse_x, cos_half_reverse_y, "cos half reverse")   

plt.title("left down")
plt.scatter(left_NF, down_NF, label = "NF")
plt.scatter(left_NM, down_NM, label = "NM")
plt.scatter(left_I, down_I, label = "I")
plt.legend()
plt.show()

plt.title("up right")
plt.scatter(up_NF, right_NF, label = "NF")
plt.scatter(up_NM, right_NM, label = "NM")
plt.scatter(up_I, right_I, label = "I")
plt.legend() 
plt.show()
 
plt.title("left circle down circle")
plt.scatter(left_circle_NF, down_circle_NF, label = "NF")
plt.scatter(left_circle_NM, down_circle_NM, label = "NM")
plt.scatter(left_circle_I, down_circle_I, label = "I")
plt.legend()
plt.show()

plt.title("up circle right circle")
plt.scatter(up_circle_NF, right_circle_NF, label = "NF")
plt.scatter(up_circle_NM, right_circle_NM, label = "NM")
plt.scatter(up_circle_I, right_circle_I, label = "I")
plt.legend() 
plt.show()