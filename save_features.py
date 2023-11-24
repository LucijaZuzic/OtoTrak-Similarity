import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np 
import pickle
from scipy.integrate import simpson
import scipy.fft
from sklearn.metrics import auc
from datetime import datetime     
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def dtw(longitudes1, latitudes1, longitudes2, latitudes2): 
    if len(longitudes1) == 0 and len(latitudes1) == 0 and len(longitudes2) == 0 and len(latitudes2) == 0:
        return 0
    if len(longitudes1) == 0 and len(latitudes1) == 0 and len(longitudes2) > 0 and len(latitudes2) > 0:
        return 10000000
    if len(longitudes1) > 0 and len(latitudes1) > 0 and len(longitudes2) == 0 and len(latitudes2) == 0:
        return 10000000 
    headlong1 = [longitudes1[0]]
    restlong1 = []
    headlong2 = [longitudes2[0]]
    restlong2 = []
    headlat1 = [latitudes1[0]]
    restlat1 = []
    headlat2 = [latitudes2[0]]
    restlat2 = []
    if len(longitudes1) > 1 and len(latitudes1) > 1:
        restlong1 = longitudes1[1:]
        restlat1 = latitudes1[1:]
    if len(longitudes2) > 1 and len(latitudes2) > 1:
        restlong2 = longitudes2[1:]
        restlat2 = latitudes2[1:]
    dtw1 = dtw(headlong1, headlat1, restlong2, restlat2) 
    dtw2 = dtw(restlong1, restlat1, restlong2, restlat2)
    dtw3 = dtw(restlong1, restlat1, headlong2, headlat2)
    return euclidean(headlong1, headlat1, headlong2, headlat2) + min(min(dtw1, dtw2), dtw3)

def decompose_fft(data: list, threshold: float = 0.0):
    fft3 = np.fft.fft(data)
    x = np.arange(0, 10, 10 / len(data))
    freqs = np.fft.fftfreq(len(x), .01)
    recomb = np.zeros((len(x),))
    for i in range(len(fft3)):
        if abs(fft3[i]) / len(x) > threshold:
            sinewave = (
                1 
                / len(x) 
                * (
                    fft3[i].real 
                    * np.cos(freqs[i] * 2 * np.pi * x) 
                    - fft3[i].imag 
                    * np.sin(freqs[i] * 2 * np.pi * x)))
            recomb += sinewave
            plt.plot(x, sinewave)
    plt.title("Sinewave")
    plt.show()

    plt.plot(x, recomb, x, data)
    plt.title("Recomb")
    plt.show()
 
def process_time(time_as_str):
    milisecond = int(time_as_str.split(".")[1])
    time_as_str = time_as_str.split(".")[0]
    epoch = datetime(1970, 1, 1)
    return (datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S') - epoch).total_seconds() + milisecond / 1000

def poly_calc(coeffs, xs):
    ys = []
    for xval in xs:
        yval = 0
        for i in range(len(coeffs)):
            yval += coeffs[i] * (xval ** (len(coeffs) - 1 - i))
        ys.append(yval)
    return ys

def get_fft_xt_yt(longitudes, latitudes, times_ride, deg): 
    xt, yt = get_poly_xt_yt(longitudes, latitudes, times_ride, deg)
    xn = poly_calc(xt, range(len(longitudes)))
    yn = poly_calc(yt, range(len(latitudes)))
    fftx = scipy.fft.fft(xn)
    ffty = scipy.fft.fft(yn)
    return xt, yt, xn, yn, fftx, ffty

def get_poly_xt_yt(longitudes, latitudes, times_ride, deg):
    xt = np.polyfit(times_ride, longitudes, deg)
    yt = np.polyfit(times_ride, latitudes, deg) 
    return xt, yt

def get_surf_xt_yt(longitudes, latitudes, times_ride, metric_used): 
    if metric_used == "trapz":
        return np.trapz(longitudes, times_ride), np.trapz(latitudes, times_ride) 
    if metric_used == "simpson":
        return simpson(longitudes, times_ride), simpson(latitudes, times_ride) 

def transform_time(times_ride): 
    times_ride = [process_time(time_as_str) for time_as_str in times_ride]
    times_ride = [time_one - times_ride[0] for time_one in times_ride] 
    return times_ride
    
def traj_len_offset(longitudes, latitudes):
    sum_dist = 0
    for i in range(len(longitudes) - 1):
        sum_dist += np.sqrt((longitudes[i + 1] - longitudes[i]) ** 2 + (latitudes[i + 1] - latitudes[i]) ** 2)
    offset_total = np.sqrt((longitudes[len(longitudes) - 1] - longitudes[0]) ** 2 + (latitudes[len(latitudes) - 1] - latitudes[0]) ** 2)
    return sum_dist, offset_total

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

def euclidean(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1)):
        sum_dist += np.sqrt((longitudes1[i] - longitudes2[i]) ** 2 + (latitudes1[i] - latitudes2[i]) ** 2)
    return sum_dist / len(longitudes1)

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
      
def scale_long_lat(long_list, lat_list, xmax = 0, ymax = 0, keep_aspect_ratio = True):
    minx = np.min(long_list)
    maxx = np.max(long_list)
    miny = np.min(lat_list)
    maxy = np.max(lat_list)
    x_diff = maxx - minx
    if x_diff == 0:
        x_diff = 1
    y_diff = maxy - miny 
    if y_diff == 0:
        y_diff = 1
    if xmax == 0 and ymax == 0 and keep_aspect_ratio:
        xmax = max(x_diff, y_diff)
        ymax = max(x_diff, y_diff)
    if xmax == 0 and ymax == 0 and not keep_aspect_ratio:
        xmax = x_diff
        ymax = y_diff
    if xmax == 0 and ymax != 0 and keep_aspect_ratio:
        xmax = ymax 
    if xmax == 0 and ymax != 0 and not keep_aspect_ratio:
        xmax = x_diff 
    if xmax != 0 and ymax == 0 and keep_aspect_ratio:
        ymax = xmax 
    if xmax != 0 and ymax == 0 and not keep_aspect_ratio:
        ymax = y_diff 
    if xmax != 0 and ymax != 0 and keep_aspect_ratio and xmax != ymax:
        ymax = xmax # ymax = xmax or xmax = ymax or keep_aspect_ratio = False or return
    long_list2 = [(x - min(long_list)) / xmax for x in long_list]
    lat_list2 = [(y - min(lat_list)) / ymax for y in lat_list]
    return long_list2, lat_list2  

def total_len(long_list, lat_list):
    total_sum = 0
    for index_coord in range(len(long_list) - 1):
        total_sum += np.sqrt((long_list[index_coord + 1] - long_list[index_coord]) ** 2 + (lat_list[index_coord + 1] - lat_list[index_coord]) ** 2)
    return total_sum

def total_offset(long_list, lat_list):
    return np.sqrt((long_list[0] - long_list[-1]) ** 2 + (lat_list[0] - lat_list[-1]) ** 2)
     
def total_angle(long_list, lat_list):
    if long_list[0] != long_list[-1]: 
        return np.arctan((lat_list[0] - lat_list[-1]) / (long_list[0] - long_list[-1]))
    else:
        return np.arctan(0)
 
def mean_vect_turning_angles(long_list, lat_list):
    total_angles = []
    for index_coord in range(len(long_list) - 1):
        if long_list[index_coord + 1] != long_list[index_coord]: 
            total_angles.append(np.arctan((lat_list[index_coord + 1] - lat_list[index_coord]) / (long_list[index_coord + 1] - long_list[index_coord]))) 
        else:
            total_angles.append(np.arctan(0))
    return np.average(total_angles)

def mean_speed_len(long_list, lat_list, times_list):
    return total_len(long_list, lat_list) / times_list[-1]

def mean_speed_offset(long_list, lat_list, times_list):
    return total_offset(long_list, lat_list) / times_list[-1]

def SomePolyArea(corners):
    n = len(corners) # of corners 
    area = 0.0
    for i in range(n):
        j = (i + 1) % n 
        area += corners[i][0] * corners[j][1] 
        area -= corners[j][0] * corners[i][1]      
    area = abs(area) / 2.0
    return area

def total_surf(long_list, lat_list):
    points_data = np.column_stack((np.array(long_list), np.array(lat_list)))  
    ch = ConvexHull(points_data)
    corners = []  
    for vx in ch.vertices:
        corners.append((points_data[vx, 0], points_data[vx, 1]))  
    value_ret = SomePolyArea(corners)
    '''
    print(value_ret)
    plt.plot(points_data[:,0], points_data[:,1], 'o')
    for vx_index in range(len(ch.vertices) - 1):
        plt.plot(points_data[[ch.vertices[vx_index], ch.vertices[vx_index + 1]], 0], points_data[[ch.vertices[vx_index], ch.vertices[vx_index + 1]], 1], 'k-')
    plt.plot(points_data[[ch.vertices[len(ch.vertices) - 1], ch.vertices[0]], 0], points_data[[ch.vertices[len(ch.vertices) - 1], ch.vertices[0]], 1], 'k-')
    '''
    return value_ret
 
window_size = 20
deg = 5
maxoffset = 0.005
step_size = window_size
#step_size = 1
max_trajs = 100
name_extension = "_window_" + str(window_size) + "_step_" + str(step_size) + "_segments_" + str(max_trajs)

all_subdirs = os.listdir() 
  
all_possible_trajs = dict()   
all_possible_trajs[window_size] = dict()

all_feats_trajs = dict()   
all_feats_trajs[window_size] = dict()

all_feats_scaled_trajs = dict()   
all_feats_scaled_trajs[window_size] = dict()

all_feats_scaled_to_max_trajs = dict()   
all_feats_scaled_to_max_trajs[window_size] = dict()

trajectory_monotonous = dict()
trajectory_monotonous[window_size] = dict()

flag_list = ["key", "flip", "zone", "engine", "in_zone", "ignition", "sleep_mode", "staff_mode", "buzzer_active", 
             "in_primary_zone", "in_restricted_zone", "onboard_geofencing", "speed_limit_active"]

trajectory_flags = dict()
for flag in flag_list:
    trajectory_flags[flag] = dict()
    trajectory_flags[flag][window_size] = dict()

label_NF = 0
label_NM = 0
label_I = 0
label_D = 0
 
total_possible_trajs = 0
 
def compare_traj_and_sample(sample_x, sample_y, sample_time, t1, metric_used): 
    if metric_used == "custom":
        return traj_dist(t1["long"], t1["lat"], sample_x, sample_y)  
    if metric_used == "dtw":
        return dtw(t1["long"], t1["lat"], sample_x, sample_y)    
    if metric_used == "trapz":
        return abs(np.trapz(t1["lat"], t1["long"]) - np.trapz(sample_y, sample_x)) 
    if metric_used == "simpson":
        return abs(simpson(t1["lat"], t1["long"]) - simpson(sample_y, sample_x))  
    if metric_used == "trapz x":
        return abs(np.trapz(t1["long"], t1["time"]) - np.trapz(sample_x, sample_time))
    if metric_used == "simpson x":
        print(t1["time"])
        return abs(simpson(t1["long"], t1["time"]) - simpson(sample_x, sample_time)) 
    if metric_used == "trapz y":
        return abs(np.trapz(t1["lat"], t1["time"]) - np.trapz(sample_y, sample_time))
    if metric_used == "simpson y": 
        return abs(simpson(t1["lat"], t1["time"]) - simpson(sample_y, sample_time))  
    if metric_used == "euclidean":
        return euclidean(t1["long"], t1["lat"], sample_x, sample_y)

metric_names = ["euclidean", "dtw", "simpson", "trapz", "custom", "simpson x", "trapz x", "simpson y", "trapz y"]
sample_names = dict()

left_edge_x = [0 for i in range(window_size)]
left_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)] 
sample_names["left"] = {"long": left_edge_x, "lat": left_edge_y}

right_edge_x = [1 for i in range(window_size)]
right_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)] 
sample_names["right"] = {"long": right_edge_x, "lat": right_edge_y}

down_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
down_edge_y = [0 for i in range(window_size)] 
sample_names["down"] = {"long": down_edge_x, "lat": down_edge_y}

up_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
up_edge_y = [1 for i in range(window_size)] 
sample_names["up"] = {"long": up_edge_x, "lat": up_edge_y}

diagonal_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
diagonal_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)] 
sample_names["diagonal"] = {"long": diagonal_edge_x, "lat": diagonal_edge_y}

left_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
left_circle_x = [np.sqrt(- y * (y - 1)) for y in left_circle_y] 
sample_names["left_circle"] = {"long": left_circle_x, "lat": left_circle_y}

right_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
right_circle_x = [1 - np.sqrt(- y * (y - 1)) for y in right_circle_y] 
sample_names["right_circle"] = {"long": right_circle_x, "lat": right_circle_y}

down_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
down_circle_y = [np.sqrt(- x * (x - 1)) for x in down_circle_x] 
sample_names["down_circle"] = {"long": down_circle_x, "lat": down_circle_y}

up_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
up_circle_y = [1 - np.sqrt(- x * (x - 1)) for x in up_circle_x]  
sample_names["up_circle"] = {"long": up_circle_x, "lat": up_circle_y}

sin_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_y = [np.sin(x * np.pi * 2) for x in sin_x]  
sample_names["sin"] = {"long": sin_x, "lat": sin_y}

sin_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_reverse_y = [np.sin(x * np.pi * 2 + np.pi) for x in sin_x]  
sample_names["sin_reverse"] = {"long": sin_reverse_x, "lat": sin_reverse_y}

sin_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_half_y = [np.sin(x * np.pi) for x in sin_half_x]  
sample_names["sin_half"] = {"long": sin_half_x, "lat": sin_half_y}

sin_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
sin_half_reverse_y = [np.sin(x * np.pi + np.pi) for x in sin_half_x]  
sample_names["sin_half_reverse"] = {"long": sin_half_reverse_x, "lat": sin_half_reverse_y}

cos_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_y = [np.cos(x * np.pi * 2) for x in cos_x]  
sample_names["cos"] = {"long": cos_x, "lat": cos_y}

cos_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_reverse_y = [np.cos(x * np.pi * 2 + np.pi) for x in cos_x]  
sample_names["cos_reverse"] = {"long": cos_reverse_x, "lat": cos_reverse_y}

cos_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_half_y = [np.cos(x * np.pi) for x in cos_half_x]  
sample_names["cos_half"] = {"long": cos_half_x, "lat": cos_half_y}

cos_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
cos_half_reverse_y = [np.cos(x * np.pi + np.pi) for x in cos_half_x]  
sample_names["cos_half_reverse"] = {"long": cos_half_reverse_x, "lat": cos_half_reverse_y}

for subdir_name in all_subdirs:

    trajs_in_dir = 0
    
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue
     
    all_possible_trajs[window_size][subdir_name] = dict() 
    all_feats_trajs[window_size][subdir_name] = dict() 
    all_feats_scaled_trajs[window_size][subdir_name] = dict() 
    all_feats_scaled_to_max_trajs[window_size][subdir_name] = dict() 
    trajectory_monotonous[window_size][subdir_name] = dict() 
    for flag in flag_list:
        trajectory_flags[flag][window_size][subdir_name] = dict() 

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
        all_feats_trajs[window_size][subdir_name][only_num_ride] = dict()
        all_feats_scaled_trajs[window_size][subdir_name][only_num_ride] = dict()
        all_feats_scaled_to_max_trajs[window_size][subdir_name][only_num_ride] = dict() 
        trajectory_monotonous[window_size][subdir_name][only_num_ride] = dict() 
        for flag in flag_list:
            trajectory_flags[flag][window_size][subdir_name][only_num_ride] = dict() 
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"]) 
        times = list(file_with_ride["time"])  
        flags_dict = dict()
        for flag in flag_list:
            flags_dict[flag] = list(file_with_ride["fields_" + flag])
  
        for x in range(0, len(longitudes) - window_size + 1, step_size):
            longitudes_tmp = longitudes[x:x + window_size]
            latitudes_tmp = latitudes[x:x + window_size]
            times_tmp = times[x:x + window_size] 
            for flag in flag_list:
                trajectory_flags_tmp = flags_dict[flag][x:x + window_size] 
 
                count_limit = False 
                for val_flag in trajectory_flags_tmp:  
                    if val_flag:
                        count_limit = True
                        break
                        
                trajectory_flags[flag][window_size][subdir_name][only_num_ride][x] = count_limit

            set_longs = set()
            set_lats = set()
            set_points = set()
            for tmp_long in longitudes_tmp:
                set_longs.add(tmp_long)
            for tmp_lat in latitudes_tmp:
                set_lats.add(tmp_lat)
            for some_index in range(len(latitudes_tmp)):
                set_points.add((latitudes_tmp[some_index], longitudes_tmp[some_index]))
                
            if len(set_lats) == 1 or len(set_longs) == 1:
                continue   
            if len(set_points) < 3:
                continue   
            
            longitudes_tmp_transform, latitudes_tmp_transform = preprocess_long_lat(longitudes_tmp, latitudes_tmp)
            
            longitudes_scaled, latitudes_scaled = scale_long_lat(longitudes_tmp_transform, latitudes_tmp_transform)
            
            longitudes_scaled_to_max, latitudes_scaled_to_max = scale_long_lat(longitudes_tmp_transform, latitudes_tmp_transform, xmax = maxoffset, ymax = maxoffset, keep_aspect_ratio = True)

            times_tmp_transform = transform_time(times_tmp)

            total_possible_trajs += 1
            trajs_in_ride += 1
            trajs_in_dir += 1 

            all_possible_trajs[window_size][subdir_name][only_num_ride][x] = {"long": longitudes_tmp_transform, "lat": latitudes_tmp_transform, "time": times_tmp_transform}

            turn_angles = mean_vect_turning_angles(longitudes_tmp_transform, latitudes_tmp_transform)  
            sp_len = mean_speed_len(longitudes_tmp_transform, latitudes_tmp_transform, times_tmp_transform)  
            sp_offset = mean_speed_offset(longitudes_tmp_transform, latitudes_tmp_transform, times_tmp_transform)   
            surfarea = total_surf(longitudes_tmp_transform, latitudes_tmp_transform) 
            surf_trapz_x, surf_trapz_y = get_surf_xt_yt(longitudes_tmp_transform, latitudes_tmp_transform, times_tmp_transform, "trapz")
            surf_simpson_x, surf_simpson_y = get_surf_xt_yt(longitudes_tmp_transform, latitudes_tmp_transform, times_tmp_transform, "simpson")
              
            turn_angles_scaled = mean_vect_turning_angles(longitudes_scaled, latitudes_scaled)  
            sp_len_scaled = mean_speed_len(longitudes_scaled, latitudes_scaled, times_tmp_transform)  
            sp_offset_scaled = mean_speed_offset(longitudes_scaled, latitudes_scaled, times_tmp_transform)   
            surfarea_scaled = total_surf(longitudes_scaled, latitudes_scaled)  
            surf_trapz_x_scaled, surf_trapz_y_scaled = get_surf_xt_yt(longitudes_scaled, latitudes_scaled, times_tmp_transform, "trapz")
            surf_simpson_x_scaled, surf_simpson_y_scaled = get_surf_xt_yt(longitudes_scaled, latitudes_scaled, times_tmp_transform, "simpson") 
        
            turn_angles_scaled_to_max = mean_vect_turning_angles(longitudes_scaled_to_max, latitudes_scaled_to_max)  
            sp_len_scaled_to_max = mean_speed_len(longitudes_scaled_to_max, latitudes_scaled_to_max, times_tmp_transform)  
            sp_offset_scaled_to_max = mean_speed_offset(longitudes_scaled_to_max, latitudes_scaled_to_max, times_tmp_transform)   
            surfarea_scaled_to_max = total_surf(longitudes_scaled_to_max, latitudes_scaled_to_max)   
            surf_trapz_x_scaled_to_max, surf_trapz_y_scaled_to_max = get_surf_xt_yt(longitudes_scaled_to_max, latitudes_scaled_to_max, times_tmp_transform, "trapz")
            surf_simpson_x_scaled_to_max, surf_simpson_y_scaled_to_max = get_surf_xt_yt(longitudes_scaled_to_max, latitudes_scaled_to_max, times_tmp_transform, "simpson") 

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

            x_poly, y_poly = get_poly_xt_yt(longitudes_tmp_transform, latitudes_tmp_transform, times_tmp_transform, deg)
            xy_poly = []
            if len(lat_sgn) == 1 and len(long_sgn) == 1:
                xy_poly = np.polyfit(longitudes_tmp_transform, latitudes_tmp_transform, deg)
                
            x_poly_scaled, y_poly_scaled = get_poly_xt_yt(longitudes_scaled, latitudes_scaled, times_tmp_transform, deg)
            xy_poly_scaled = []
            if len(lat_sgn) == 1 and len(long_sgn) == 1:
                xy_poly_scaled = np.polyfit(longitudes_scaled, latitudes_scaled, deg)

            x_poly_scaled_to_max, y_poly_scaled_to_max = get_poly_xt_yt(longitudes_scaled_to_max, latitudes_scaled_to_max, times_tmp_transform, deg)
            xy_poly_scaled_to_max = []
            if len(lat_sgn) == 1 and len(long_sgn) == 1:
                xy_poly_scaled_to_max = np.polyfit(longitudes_scaled_to_max, latitudes_scaled_to_max, deg)

            all_feats_trajs[window_size][subdir_name][only_num_ride][x] = {"mean_vect_turning_angles": turn_angles / np.pi * 180, 
                                                                           "max_x": max(longitudes_tmp_transform),
                                                                           "max_y": max(latitudes_tmp_transform),
                                                                           "surf_trapz_x": surf_trapz_x, 
                                                                           "surf_trapz_y": surf_trapz_y, 
                                                                           "surf_simpson_x": surf_simpson_x, 
                                                                           "surf_simpson_y": surf_simpson_y, 
                                                                           "x_poly": x_poly, 
                                                                           "y_poly": y_poly, 
                                                                           "xy_poly": xy_poly, 
                                                                           "duration": times_tmp_transform[-1],
                                                                           "len": sp_len * times_tmp_transform[-1], 
                                                                           "offset": sp_offset * times_tmp_transform[-1],
                                                                           "mean_speed_len": sp_len, 
                                                                           "mean_speed_offset": sp_offset,
                                                                           "len_vs_offset": sp_len / sp_offset,
                                                                           "total_surf": surfarea}

            for sample_name in sample_names:
                 for metric_name in metric_names: 
                    oldx = [valx for valx in sample_names[sample_name]["long"]]
                    oldy = [valy for valy in sample_names[sample_name]["lat"]]
                    newx = [valx * max(max(longitudes_tmp_transform), max(latitudes_tmp_transform)) for valx in sample_names[sample_name]["long"]]
                    newy = [valy * max(max(longitudes_tmp_transform), max(latitudes_tmp_transform)) for valy in sample_names[sample_name]["lat"]]
                    all_feats_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_same_" + metric_name] = compare_traj_and_sample(newx, newy, range(len(newx)), {"long": longitudes_tmp_transform, "lat": latitudes_tmp_transform, "time": times_tmp_transform}, metric_name)
                    all_feats_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_diff_" + metric_name] = compare_traj_and_sample(oldx, oldy, range(len(oldx)), {"long": longitudes_tmp_transform, "lat": latitudes_tmp_transform, "time": times_tmp_transform}, metric_name)

            all_feats_scaled_trajs[window_size][subdir_name][only_num_ride][x] = {"mean_vect_turning_angles": turn_angles_scaled / np.pi * 180, 
                                                                           "max_x": max(longitudes_scaled),
                                                                           "max_y": max(latitudes_scaled),
                                                                           "surf_trapz_x": surf_trapz_x_scaled, 
                                                                           "surf_trapz_y": surf_trapz_y_scaled, 
                                                                           "surf_simpson_x": surf_simpson_x_scaled, 
                                                                           "surf_simpson_y": surf_simpson_y_scaled, 
                                                                           "x_poly": x_poly_scaled, 
                                                                           "y_poly": y_poly_scaled, 
                                                                           "xy_poly": xy_poly_scaled, 
                                                                           "duration": times_tmp_transform[-1],
                                                                           "len": sp_len_scaled * times_tmp_transform[-1], 
                                                                           "offset": sp_offset_scaled * times_tmp_transform[-1],
                                                                           "mean_speed_len": sp_len_scaled, 
                                                                           "mean_speed_offset": sp_offset_scaled,
                                                                           "len_vs_offset": sp_len_scaled / sp_offset_scaled,
                                                                           "total_surf": surfarea_scaled} 
            
            for sample_name in sample_names:
                 for metric_name in metric_names: 
                    oldx = [valx for valx in sample_names[sample_name]["long"]]
                    oldy = [valy for valy in sample_names[sample_name]["lat"]]
                    newx = [valx * max(max(longitudes_scaled), max(latitudes_scaled)) for valx in sample_names[sample_name]["long"]]
                    newy = [valy * max(max(longitudes_scaled), max(latitudes_scaled)) for valy in sample_names[sample_name]["lat"]]
                    all_feats_scaled_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_same_" + metric_name] = compare_traj_and_sample(newx, newy, range(len(newx)), {"long": longitudes_scaled, "lat": latitudes_scaled, "time": times_tmp_transform}, metric_name)
                    all_feats_scaled_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_diff_" + metric_name] = compare_traj_and_sample(oldx, oldy, range(len(oldx)), {"long": longitudes_scaled, "lat": latitudes_scaled, "time": times_tmp_transform}, metric_name)

            all_feats_scaled_to_max_trajs[window_size][subdir_name][only_num_ride][x] = {"mean_vect_turning_angles": turn_angles_scaled_to_max / np.pi * 180, 
                                                                           "max_x": max(longitudes_scaled_to_max),
                                                                           "max_y": max(latitudes_scaled_to_max),
                                                                           "surf_trapz_x": surf_trapz_x_scaled_to_max, 
                                                                           "surf_trapz_y": surf_trapz_y_scaled_to_max, 
                                                                           "surf_simpson_x": surf_simpson_x_scaled_to_max, 
                                                                           "surf_simpson_y": surf_simpson_y_scaled_to_max, 
                                                                           "x_poly": x_poly_scaled_to_max, 
                                                                           "y_poly": y_poly_scaled_to_max, 
                                                                           "xy_poly": xy_poly_scaled_to_max, 
                                                                           "duration": times_tmp_transform[-1],
                                                                           "len": sp_len_scaled_to_max * times_tmp_transform[-1], 
                                                                           "offset": sp_offset_scaled_to_max * times_tmp_transform[-1],
                                                                           "mean_speed_len": sp_len_scaled_to_max, 
                                                                           "mean_speed_offset": sp_offset_scaled_to_max,
                                                                           "len_vs_offset": sp_len_scaled_to_max / sp_offset_scaled_to_max,
                                                                           "total_surf": surfarea_scaled_to_max}
            
            for sample_name in sample_names:
                 for metric_name in metric_names: 
                    oldx = [valx for valx in sample_names[sample_name]["long"]]
                    oldy = [valy for valy in sample_names[sample_name]["lat"]]
                    newx = [valx * max(max(longitudes_scaled_to_max), max(latitudes_scaled_to_max)) for valx in sample_names[sample_name]["long"]]
                    newy = [valy * max(max(longitudes_scaled_to_max), max(latitudes_scaled_to_max)) for valy in sample_names[sample_name]["lat"]]
                    all_feats_scaled_to_max_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_same_" + metric_name] = compare_traj_and_sample(newx, newy, range(len(newx)), {"long": longitudes_scaled_to_max, "lat": latitudes_scaled_to_max, "time": times_tmp_transform}, metric_name)
                    all_feats_scaled_to_max_trajs[window_size][subdir_name][only_num_ride][x][sample_name + "_diff_" + metric_name] = compare_traj_and_sample(oldx, oldy, range(len(oldx)), {"long": longitudes_scaled_to_max, "lat": latitudes_scaled_to_max, "time": times_tmp_transform}, metric_name)
            
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
print("NF", label_NF, "NM", label_NM, "D", label_D, "I", label_I) 

def process_csv(some_dict, save_name): 
    new_csv_content = "window_size,vehicle,ride,start,mean_vect_turning_angles,max_x,max_y,surf_trapz_x,surf_trapz_y,surf_simpson_x,surf_simpson_y,"
    for d in range(deg + 1):
        new_csv_content += "x_poly_" + str(d + 1) + ","
    for d in range(deg + 1):
        new_csv_content += "y_poly_" + str(d + 1) + ","
    for d in range(deg + 1):
        new_csv_content += "xy_poly_" + str(d + 1) + "," 
    new_csv_content += "duration,len,offset,mean_speed_len,mean_speed_offset,len_vs_offset,total_surf,"
    for sample_name in sample_names:
        for metric_name in metric_names: 
            new_csv_content += sample_name + "_same_" + metric_name + "," 
            new_csv_content += sample_name + "_diff_" + metric_name + ","
    new_csv_content += "monotonous," 
    for flag in flag_list:
        new_csv_content += flag + ","
    new_csv_content += "\n"
    for vehicle1 in all_possible_trajs[window_size].keys():  
        for r1 in all_possible_trajs[window_size][vehicle1]:
            for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
                new_csv_content += str(window_size) + "," + str(vehicle1) + "," + str(r1) + "," + str(x1) + ","  
                for feat_name in some_dict[window_size][vehicle1][r1][x1]: 
                    if "poly" in feat_name: 
                        for val in list(some_dict[window_size][vehicle1][r1][x1][feat_name]): 
                            new_csv_content += str(val) + ","
                        if len(list(some_dict[window_size][vehicle1][r1][x1][feat_name])) == 0:
                            for d in range(deg + 1): 
                                new_csv_content += ","
                    else:
                        new_csv_content += str(some_dict[window_size][vehicle1][r1][x1][feat_name]) + ","
                new_csv_content += trajectory_monotonous[window_size][vehicle1][r1][x1] + ","  
                for flag in flag_list:
                     new_csv_content += str(trajectory_flags[flag][window_size][vehicle1][r1][x1]) + ","  
                new_csv_content += "\n"    
    csv_file = open(save_name, "w")
    csv_file.write(new_csv_content)
    csv_file.close()
  
process_csv(all_feats_trajs, "all_feats.csv")
process_csv(all_feats_scaled_trajs, "all_feats_scaled.csv")
process_csv(all_feats_scaled_to_max_trajs, "all_feats_scaled_to_max.csv")
