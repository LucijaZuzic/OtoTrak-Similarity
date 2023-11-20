import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np 
import pickle
from scipy.integrate import simpson
import scipy.fft
from sklearn.metrics import auc
from datetime import datetime   

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
  time_as_str = time_as_str.split(".")[0]
  return datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S')

def poly_calc(coeffs, xs):
  ys = []
  for xval in xs:
    yval = 0
    for i in range(len(coeffs)):
      yval += coeffs[i] * (xval ** (len(coeffs) - 1 - i))
    ys.append(yval)
  return ys

def get_fft_xt_yt(longitudes, latitudes, times_ride): 
  xt, yt = get_poly_xt_yt(longitudes, latitudes, times_ride)
  xn = poly_calc(xt, range(len(longitudes)))
  yn = poly_calc(yt, range(len(latitudes)))
  fftx = scipy.fft.fft(xn)
  ffty = scipy.fft.fft(yn)
  return xt, yt, xn, yn, fftx, ffty

def get_poly_xt_yt(longitudes, latitudes, times_ride):
  xt = np.polyfit(times_ride, longitudes, len(longitudes))
  yt = np.polyfit(times_ride, latitudes, len(latitudes)) 
  return xt, yt

def get_surf_xt_yt(longitudes, latitudes, times_ride, metric_used): 
  if metric_used == "trapz":
    return np.trapz(longitudes, times_ride), np.trapz(latitudes, times_ride) 
  if metric_used == "simpson":
    return simpson(longitudes, times_ride), simpson(latitudes, times_ride) 

def transform_time(times_ride): 
  times_ride = [process_time(time_as_str) for time_as_str in times_ride]
  times_ride = [(time_one - times_ride[0]).total_seconds() for time_one in times_ride] 
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

trajectory_sl = dict()
trajectory_sl[window_size] = dict()

label_NF = 0
label_NM = 0
label_I = 0
label_D = 0

label_T_NF = 0
label_T_NM = 0
label_T_I = 0
label_T_D = 0
 
total_possible_trajs = 0

for subdir_name in all_subdirs:

  trajs_in_dir = 0
  
  if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
    continue
   
  all_possible_trajs[window_size][subdir_name] = dict() 
  trajectory_monotonous[window_size][subdir_name] = dict() 
  trajectory_sl[window_size][subdir_name] = dict() 

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
    trajectory_sl[window_size][subdir_name][only_num_ride] = dict()
  
    file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
    longitudes = list(file_with_ride["fields_longitude"])
    latitudes = list(file_with_ride["fields_latitude"]) 
    times = list(file_with_ride["time"]) 
    spl = list(file_with_ride["fields_speed_limit_active"]) 
 
    for x in range(0, len(longitudes) - window_size + 1, step_size):
      longitudes_tmp = longitudes[x:x + window_size]
      latitudes_tmp = latitudes[x:x + window_size]
      times_tmp = times[x:x + window_size]
      sl_tmp = spl[x:x + window_size]

      exceed_sl = False
      for sl in sl_tmp:
        if sl:
          exceed_sl = True
          break

      trajectory_sl[window_size][subdir_name][only_num_ride][x] = exceed_sl

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

      times_tmp_transform = transform_time(times_tmp)

      total_possible_trajs += 1
      trajs_in_ride += 1
      trajs_in_dir += 1 

      all_possible_trajs[window_size][subdir_name][only_num_ride][x] = {"long": longitudes_tmp_transform, "lat": latitudes_tmp_transform, "time": times_tmp_transform}

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
        if exceed_sl:
          label_T_NF += 1
      if (len(lat_sgn) == 1 and len(long_sgn) > 1) or (len(lat_sgn) > 1 and len(long_sgn) == 1):
        trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "NM"
        label_NM += 1
        if exceed_sl:
          label_T_NM += 1
      if len(lat_sgn) == 1 and len(long_sgn) == 1:
        if (True in lat_sgn and True in long_sgn) or (False in lat_sgn and False in long_sgn):
          trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "I"
          label_D += 1
          if exceed_sl:
            label_T_D += 1
        else:
          trajectory_monotonous[window_size][subdir_name][only_num_ride][x] = "D"
          label_I += 1
          if exceed_sl:
            label_T_I += 1
         
    #print(only_num_ride, trajs_in_ride)
  print(subdir_name, trajs_in_dir)
print(total_possible_trajs)
print("NF", label_NF, "NM", label_NM, "D", label_D, "I", label_I) 
print("NF T", label_T_NF, "NM T", label_T_NM, "D T", label_T_D, "I T", label_T_I) 
print("NF F", label_NF - label_T_NF, "NM F", label_NM - label_T_NM, "D F", label_D - label_T_D, "I F", label_I - label_T_I) 

def compare_all_with_sample(metric_used, sample_x, sample_y, title_sample):
 
  distances_from_sample_ride_vehicle = [] 
  labels_from_sample_ride_vehicle = [] 
  sl_from_sample_ride_vehicle = [] 

  max_distances_from_sample_ride_vehicle = 0
  index_max_distances_from_sample_ride_vehicle = 0 

  min_distances_from_sample_ride_vehicle = 100000
  index_min_distances_from_sample_ride_vehicle = 0 

  for vehicle1 in all_possible_trajs[window_size].keys(): 
    for r1 in all_possible_trajs[window_size][vehicle1]: 
      for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
        t1 = all_possible_trajs[window_size][vehicle1][r1][x1]  
        #td_up_auc = abs(auc(t1["long"], t1["lat"]) - auc(sample_x, sample_y)) - isti kao trapz, x mora biti rastući ili padajući
        #print(td_up, td_up_trapz, td_up_simpson)#, td_up_auc)
        if metric_used == "custom":
          td_up = traj_dist(t1["long"], t1["lat"], sample_x, sample_y) 
          distances_from_sample_ride_vehicle.append(td_up) 
        if metric_used == "dtw":
          td_up_dtw = dtw(t1["long"], t1["lat"], sample_x, sample_y) 
          distances_from_sample_ride_vehicle.append(td_up_dtw) 
        if metric_used == "trapz":
          td_up_trapz = abs(np.trapz(t1["lat"], t1["long"]) - np.trapz(sample_y, sample_x))
          distances_from_sample_ride_vehicle.append(td_up_trapz) 
        if metric_used == "simpson":
          td_up_simpson = abs(simpson(t1["lat"], t1["long"]) - simpson(sample_y, sample_x))
          distances_from_sample_ride_vehicle.append(td_up_simpson) 
        if metric_used == "euclidean":
          td_up_euclidean = euclidean(t1["long"], t1["lat"], sample_x, sample_y)
          distances_from_sample_ride_vehicle.append(td_up_euclidean) 
        labels_from_sample_ride_vehicle.append(trajectory_monotonous[window_size][vehicle1][r1][x1])
        sl_from_sample_ride_vehicle.append(trajectory_sl[window_size][vehicle1][r1][x1])

        if distances_from_sample_ride_vehicle[-1] > max_distances_from_sample_ride_vehicle:
          index_max_distances_from_sample_ride_vehicle = t1
          max_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]
 
        if distances_from_sample_ride_vehicle[-1] < min_distances_from_sample_ride_vehicle:
          index_min_distances_from_sample_ride_vehicle = t1
          min_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]

  plt.subplot(1, 2, 1)
  plt.title("Min " + metric_used + " " + title_sample)
  plt.plot(index_min_distances_from_sample_ride_vehicle["long"], index_min_distances_from_sample_ride_vehicle["lat"])
  plt.plot(sample_x, sample_y)
  plt.subplot(1, 2, 2)
  plt.title("Max " + metric_used + " " + title_sample)
  plt.plot(index_max_distances_from_sample_ride_vehicle["long"], index_max_distances_from_sample_ride_vehicle["lat"])
  plt.plot(sample_x, sample_y)
  plt.savefig("Min Max " + metric_used + " " + title_sample + ".png", bbox_inches = "tight")
  plt.close()
  
  only_NF = []
  only_NM = []
  only_I = []
  only_D = []
  only_NF_T = []
  only_NM_T = []
  only_I_T = []
  only_D_T = []
  only_NF_F = []
  only_NM_F = []
  only_I_F = []
  only_D_F = []
  only_T = []
  only_F = []
  for i in range(len(distances_from_sample_ride_vehicle)):
    if sl_from_sample_ride_vehicle[i]:
      only_T.append(distances_from_sample_ride_vehicle[i])
    else:
      only_F.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i] == "NF":
      only_NF.append(distances_from_sample_ride_vehicle[i])
      if sl_from_sample_ride_vehicle[i]:
        only_NF_T.append(distances_from_sample_ride_vehicle[i])
      else:
        only_NF_F.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i] == "NM":
      only_NM.append(distances_from_sample_ride_vehicle[i])
      if sl_from_sample_ride_vehicle[i]:
        only_NM_T.append(distances_from_sample_ride_vehicle[i])
      else:
        only_NM_F.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i] == "I":
      only_I.append(distances_from_sample_ride_vehicle[i])
      if sl_from_sample_ride_vehicle[i]:
        only_I_T.append(distances_from_sample_ride_vehicle[i])
      else:
        only_I_F.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i] == "D":
      only_D.append(distances_from_sample_ride_vehicle[i])
      if sl_from_sample_ride_vehicle[i]:
        only_D_T.append(distances_from_sample_ride_vehicle[i])
      else:
        only_D_F.append(distances_from_sample_ride_vehicle[i])

  plt.subplot(2, 2, 1)
  plt.title("Dist " + metric_used + " " + title_sample)
  plt.hist(distances_from_sample_ride_vehicle, label = "All") 
  plt.hist(only_F, label = "F") 
  plt.hist(only_T, label = "T") 
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Dist NF " + metric_used + " " + title_sample)
  plt.hist(only_NF, label = "NM") 
  plt.hist(only_NF_F, label = "NM F") 
  plt.hist(only_NF_T, label = "NM T") 
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Dist NM " + metric_used + " " + title_sample)
  plt.hist(only_NM, label = "NM") 
  plt.hist(only_NM_F, label = "NM F") 
  plt.hist(only_NM_T, label = "NM T") 
  plt.legend()
  plt.subplot(2, 2, 4)
  plt.title("Dist I " + metric_used + " " + title_sample)
  plt.hist(only_I, label = "I") 
  plt.hist(only_I_F, label = "I F") 
  plt.hist(only_I_T, label = "I T") 
  plt.legend()
  plt.savefig("Dist " + metric_used + " " + title_sample + ".png", bbox_inches = "tight")
  plt.close()
  
  return distances_from_sample_ride_vehicle, labels_from_sample_ride_vehicle, only_NF, only_NM, only_I, only_NF_T, only_NM_T, only_I_T, only_NF_F, only_NM_F, only_I_F

def use_metric(metric_used): 
  left_edge_x = [0 for i in range(window_size)]
  left_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
  left_dist, traj_labels, left_NF, left_NM, left_I, left_NF_T, left_NM_T, left_I_T, left_NF_F, left_NM_F, left_I_F = compare_all_with_sample(metric_used, left_edge_x, left_edge_y, "left")

  right_edge_x = [1 for i in range(window_size)]
  right_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
  right_dist, traj_labels, right_NF, right_NM, right_I, right_NF_T, right_NM_T, right_I_T, right_NF_F, right_NM_F, right_I_F = compare_all_with_sample(metric_used, right_edge_x, right_edge_y, "right")

  down_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  down_edge_y = [0 for i in range(window_size)]
  down_dist, traj_labels, down_NF, down_NM, down_I, down_NF_T, down_NM_T, down_I_T, down_NF_F, down_NM_F, down_I_F = compare_all_with_sample(metric_used, down_edge_x, down_edge_y, "down")

  up_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  up_edge_y = [1 for i in range(window_size)]
  up_dist, traj_labels, up_NF, up_NM, up_I, up_NF_T, up_NM_T, up_I_T, up_NF_F, up_NM_F, up_I_F = compare_all_with_sample(metric_used, up_edge_x, up_edge_y, "up")

  diagonal_edge_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  diagonal_edge_y = [x * 1 / (window_size - 1) for x in range(window_size)]
  diagonal_dist, traj_labels, diagonal_NF, diagonal_NM, diagonal_I, diagonal_NF_T, diagonal_NM_T, diagonal_I_T, diagonal_NF_F, diagonal_NM_F, diagonal_I_F = compare_all_with_sample(metric_used, diagonal_edge_x, diagonal_edge_y, "diagonal")

  left_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
  left_circle_x = [np.sqrt(- y * (y - 1)) for y in left_circle_y]
  left_circle_dist, traj_labels, left_circle_NF, left_circle_NM, left_circle_I, left_circle_NF_T, left_circle_NM_T, left_circle_I_T, left_circle_NF_F, left_circle_NM_F, left_circle_I_F = compare_all_with_sample(metric_used, left_circle_x, left_circle_y, "left_circle")

  right_circle_y = [x * 1 / (window_size - 1) for x in range(window_size)]
  right_circle_x = [1 - np.sqrt(- y * (y - 1)) for y in right_circle_y]
  right_circle_dist, traj_labels, right_circle_NF, right_circle_NM, right_circle_I, right_circle_NF_T, right_circle_NM_T, right_circle_I_T, right_circle_NF_F, right_circle_NM_F, right_circle_I_F = compare_all_with_sample(metric_used, right_circle_x, right_circle_y, "right_circle")

  down_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  down_circle_y = [np.sqrt(- x * (x - 1)) for x in down_circle_x]
  down_circle_dist, traj_labels, down_circle_NF, down_circle_NM, down_circle_I, down_circle_NF_T, down_circle_NM_T, down_circle_I_T, down_circle_NF_F, down_circle_NM_F, down_circle_I_F = compare_all_with_sample(metric_used, down_circle_x, down_circle_y, "down_circle")

  up_circle_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  up_circle_y = [1 - np.sqrt(- x * (x - 1)) for x in up_circle_x] 
  up_circle_dist, traj_labels, up_circle_NF, up_circle_NM, up_circle_I, up_circle_NF_T, up_circle_NM_T, up_circle_I_T, up_circle_NF_F, up_circle_NM_F, up_circle_I_F = compare_all_with_sample(metric_used, up_circle_x, up_circle_y, "up_circle")
  
  sin_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  sin_y = [np.sin(x * np.pi * 2) for x in sin_x] 
  sin_dist, traj_labels, sin_NF, sin_NM, sin_I, sin_NF_T, sin_NM_T, sin_I_T, sin_NF_F, sin_NM_F, sin_I_F = compare_all_with_sample(metric_used, sin_x, sin_y, "sin")

  sin_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  sin_reverse_y = [np.sin(x * np.pi * 2 + np.pi) for x in sin_x] 
  sin_reverse_dist, traj_labels, sin_reverse_NF, sin_reverse_NM, sin_reverse_I, sin_reverse_NF_T, sin_reverse_NM_T, sin_reverse_I_T, sin_reverse_NF_F, sin_reverse_NM_F, sin_reverse_I_F = compare_all_with_sample(metric_used, sin_reverse_x, sin_reverse_y, "sin_reverse")

  sin_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  sin_half_y = [np.sin(x * np.pi) for x in sin_half_x] 
  sin_half_dist, traj_labels, sin_half_NF, sin_half_NM, sin_half_I, sin_half_NF_T, sin_half_NM_T, sin_half_I_T, sin_half_NF_F, sin_half_NM_F, sin_half_I_F = compare_all_with_sample(metric_used, sin_half_x, sin_half_y, "sin_half")

  sin_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  sin_half_reverse_y = [np.sin(x * np.pi + np.pi) for x in sin_half_x] 
  sin_half_reverse_dist, traj_labels, sin_half_reverse_NF, sin_half_reverse_NM, sin_half_reverse_I, sin_half_reverse_NF_T, sin_half_reverse_NM_T, sin_half_reverse_I_T, sin_half_reverse_NF_F, sin_half_reverse_NM_F, sin_half_reverse_I_F = compare_all_with_sample(metric_used, sin_half_reverse_x, sin_half_reverse_y, "sin_half_reverse")
  
  cos_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  cos_y = [np.cos(x * np.pi * 2) for x in cos_x] 
  cos_dist, traj_labels, cos_NF, cos_NM, cos_I, cos_NF_T, cos_NM_T, cos_I_T, cos_NF_F, cos_NM_F, cos_I_F = compare_all_with_sample(metric_used, cos_x, cos_y, "cos")

  cos_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  cos_reverse_y = [np.cos(x * np.pi * 2 + np.pi) for x in cos_x] 
  cos_reverse_dist, traj_labels, cos_reverse_NF, cos_reverse_NM, cos_reverse_I, cos_reverse_NF_T, cos_reverse_NM_T, cos_reverse_I_T, cos_reverse_NF_F, cos_reverse_NM_F, cos_reverse_I_F = compare_all_with_sample(metric_used, cos_reverse_x, cos_reverse_y, "cos_reverse")

  cos_half_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  cos_half_y = [np.cos(x * np.pi) for x in cos_half_x] 
  cos_half_dist, traj_labels, cos_half_NF, cos_half_NM, cos_half_I, cos_half_NF_T, cos_half_NM_T, cos_half_I_T, cos_half_NF_F, cos_half_NM_F, cos_half_I_F = compare_all_with_sample(metric_used, cos_half_x, cos_half_y, "cos_half")

  cos_half_reverse_x = [x * 1 / (window_size - 1) for x in range(window_size)]
  cos_half_reverse_y = [np.cos(x * np.pi + np.pi) for x in cos_half_x] 
  cos_half_reverse_dist, traj_labels, cos_half_reverse_NF, cos_half_reverse_NM, cos_half_reverse_I, cos_half_reverse_NF_T, cos_half_reverse_NM_T, cos_half_reverse_I_T, cos_half_reverse_NF_F, cos_half_reverse_NM_F, cos_half_reverse_I_F = compare_all_with_sample(metric_used, cos_half_reverse_x, cos_half_reverse_y, "cos_half_reverse")
  
  plt.subplot(2, 2, 1)
  plt.title("Compare left down " + metric_used)
  plt.scatter(left_NF + left_NM + left_I, down_NF + down_NM + down_I, label = "All")
  plt.scatter(left_NF_T + left_NM_T + left_I_T, down_NF_T + down_NM_T + down_I_T, label = "T")
  plt.scatter(left_NF_F + left_NM_F + left_I_F, down_NF_F + down_NM_F + down_I_F, label = "F") 
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Compare left down NF" + metric_used)
  plt.scatter(left_NF, down_NF, label = "NF")
  plt.scatter(left_NF_T, down_NF_T, label = "NF T")
  plt.scatter(left_NF_F, down_NF_F, label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Compare left down NM " + metric_used)
  plt.scatter(left_NM, down_NM, label = "NM")
  plt.scatter(left_NM_T, down_NM_T, label = "NM T")
  plt.scatter(left_NM_F, down_NM_F, label = "NM F")
  plt.legend() 
  plt.subplot(2, 2, 4)
  plt.title("Compare left down I " + metric_used)
  plt.scatter(left_I, down_I, label = "I")
  plt.scatter(left_I_T, down_I_T, label = "I T")
  plt.scatter(left_I_F, down_I_F, label = "I F")
  plt.legend()
  plt.savefig("Compare left down " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  plt.subplot(2, 2, 1)
  plt.title("Compare right up " + metric_used)
  plt.scatter(right_NF + right_NM + right_I, up_NF + up_NM + up_I, label = "All")
  plt.scatter(right_NF_T + right_NM_T + right_I_T, up_NF_T + up_NM_T + up_I_T, label = "T")
  plt.scatter(right_NF_F + right_NM_F + right_I_F, up_NF_F + up_NM_F + up_I_F, label = "F") 
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Compare right up NF" + metric_used)
  plt.scatter(right_NF, up_NF, label = "NF")
  plt.scatter(right_NF_T, up_NF_T, label = "NF T")
  plt.scatter(right_NF_F, up_NF_F, label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Compare right up NM " + metric_used)
  plt.scatter(right_NM, up_NM, label = "NM")
  plt.scatter(right_NM_T, up_NM_T, label = "NM T")
  plt.scatter(right_NM_F, up_NM_F, label = "NM F")
  plt.legend() 
  plt.subplot(2, 2, 4)
  plt.title("Compare right up I " + metric_used)
  plt.scatter(right_I, up_I, label = "I")
  plt.scatter(right_I_T, up_I_T, label = "I T")
  plt.scatter(right_I_F, up_I_F, label = "I F")
  plt.legend()
  plt.savefig("Compare right up " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  plt.subplot(2, 2, 1)
  plt.title("Compare left circle down circle " + metric_used)  
  plt.scatter(left_circle_NF + left_circle_NM + left_circle_I, down_circle_NF + down_circle_NM + down_circle_I, label = "All")
  plt.scatter(left_circle_NF_T + left_circle_NM_T + left_circle_I_T, down_circle_NF_T + down_circle_NM_T + down_circle_I_T, label = "T")
  plt.scatter(left_circle_NF_F + left_circle_NM_F + left_circle_I_F, down_circle_NF_F + down_circle_NM_F + down_circle_I_F, label = "F") 
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Compare left circle down circle NF" + metric_used)
  plt.scatter(left_circle_NF, down_circle_NF, label = "NF")
  plt.scatter(left_circle_NF_T, down_circle_NF_T, label = "NF T")
  plt.scatter(left_circle_NF_F, down_circle_NF_F, label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Compare left circle down circle NM " + metric_used)
  plt.scatter(left_circle_NM, down_circle_NM, label = "NM")
  plt.scatter(left_circle_NM_T, down_circle_NM_T, label = "NM T")
  plt.scatter(left_circle_NM_F, down_circle_NM_F, label = "NM F")
  plt.legend() 
  plt.subplot(2, 2, 4)
  plt.title("Compare left circle down circle I " + metric_used)
  plt.scatter(left_circle_I, down_circle_I, label = "I")
  plt.scatter(left_circle_I_T, down_circle_I_T, label = "I T")
  plt.scatter(left_circle_I_F, down_circle_I_F, label = "I F")
  plt.legend()
  plt.savefig("Compare left circle down circle " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  plt.subplot(2, 2, 1)
  plt.title("Compare right circle up circle " + metric_used)
  plt.scatter(right_circle_NF + right_circle_NM + right_circle_I, up_circle_NF + up_circle_NM + up_circle_I, label = "All")
  plt.scatter(right_circle_NF_T + right_circle_NM_T + right_circle_I_T, up_circle_NF_T + up_circle_NM_T + up_circle_I_T, label = "T")
  plt.scatter(right_circle_NF_F + right_circle_NM_F + right_circle_I_F, up_circle_NF_F + up_circle_NM_F + up_circle_I_F, label = "F") 
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Compare right circle up circle NF" + metric_used)
  plt.scatter(right_circle_NF, up_circle_NF, label = "NF")
  plt.scatter(right_circle_NF_T, up_circle_NF_T, label = "NF T")
  plt.scatter(right_circle_NF_F, up_circle_NF_F, label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Compare right circle up circle NM " + metric_used)
  plt.scatter(right_circle_NM, up_circle_NM, label = "NM")
  plt.scatter(right_circle_NM_T, up_circle_NM_T, label = "NM T")
  plt.scatter(right_circle_NM_F, up_circle_NM_F, label = "NM F")
  plt.legend() 
  plt.subplot(2, 2, 4)
  plt.title("Compare right circle up circle I " + metric_used)
  plt.scatter(right_circle_I, up_circle_I, label = "I")
  plt.scatter(right_circle_I_T, up_circle_I_T, label = "I T")
  plt.scatter(right_circle_I_F, up_circle_I_F, label = "I F")
  plt.legend()
  plt.savefig("Compare right circle up circle " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

use_metric("custom")
use_metric("dtw")
use_metric("simpson")
use_metric("trapz")
use_metric("euclidean")

def compare_all(metric_used):
 
  distances_from_sample_ride_vehicle = [] 
  trajs_from_sample_ride_vehicle = [] 
  labels_from_sample_ride_vehicle = [] 

  max_distances_from_sample_ride_vehicle = 0
  index_max_distances_from_sample_ride_vehicle_1 = 0 
  index_max_distances_from_sample_ride_vehicle_2 = 0 

  min_distances_from_sample_ride_vehicle = 100000
  index_min_distances_from_sample_ride_vehicle_1 = 0 
  index_min_distances_from_sample_ride_vehicle_2 = 0 

  for vehicle1 in all_possible_trajs[window_size].keys(): 
    for r1 in all_possible_trajs[window_size][vehicle1]: 
      for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
        t1 = all_possible_trajs[window_size][vehicle1][r1][x1] 
        for vehicle2 in all_possible_trajs[window_size].keys():
          #if vehicle2 == vehicle1:
            #continue  
          for r2 in all_possible_trajs[window_size][vehicle2]: 
            for x2 in all_possible_trajs[window_size][vehicle2][r2]: 
              if vehicle2 == vehicle1 and r1 == r2 and x1 == x2:
                continue  
              t2 = all_possible_trajs[window_size][vehicle2][r2][x2]
              #td_up_auc = abs(auc(t1["long"], t1["lat"]) - auc(sample_x, sample_y)) - isti kao trapz, x mora biti rastući ili padajući
              #print(td_up, td_up_trapz, td_up_simpson)#, td_up_auc)
              if metric_used == "custom":
                td_up = traj_dist(t1["long"], t1["lat"], t2["long"], t2["lat"])
                distances_from_sample_ride_vehicle.append(td_up)
              if metric_used == "dtw":
                td_up_dtw = dtw(t1["long"], t1["lat"], t2["long"], t2["lat"]) 
                distances_from_sample_ride_vehicle.append(td_up_dtw)  
              if metric_used == "trapz":
                td_up_trapz = abs(np.trapz(t1["lat"], t1["long"]) - np.trapz(t2["lat"], t2["long"]))
                distances_from_sample_ride_vehicle.append(td_up_trapz) 
              if metric_used == "simpson":
                td_up_simpson = abs(simpson(t1["lat"], t1["long"]) - simpson(t2["lat"], t2["long"]))
                distances_from_sample_ride_vehicle.append(td_up_simpson) 
              if metric_used == "euclidean":
                td_up_euclidean = euclidean(t1["long"], t1["lat"], t2["long"], t2["lat"])
                distances_from_sample_ride_vehicle.append(td_up_euclidean) 
              trajs_from_sample_ride_vehicle.append((t1, t2))
              labels_from_sample_ride_vehicle.append((trajectory_monotonous[window_size][vehicle1][r1][x1], trajectory_monotonous[window_size][vehicle2][r2][x2]))

              if distances_from_sample_ride_vehicle[-1] > max_distances_from_sample_ride_vehicle:
                index_max_distances_from_sample_ride_vehicle_1 = t1
                index_max_distances_from_sample_ride_vehicle_2 = t2
                max_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]
      
              if distances_from_sample_ride_vehicle[-1] < min_distances_from_sample_ride_vehicle:
                index_min_distances_from_sample_ride_vehicle_1 = t1
                index_min_distances_from_sample_ride_vehicle_2 = t2
                min_distances_from_sample_ride_vehicle = distances_from_sample_ride_vehicle[-1]

  plt.subplot(1, 2, 1)
  plt.title("Min " + metric_used)
  plt.plot(index_min_distances_from_sample_ride_vehicle_1["long"], index_min_distances_from_sample_ride_vehicle_1["lat"])
  plt.plot(index_min_distances_from_sample_ride_vehicle_2["long"], index_min_distances_from_sample_ride_vehicle_2["lat"]) 
  plt.subplot(1, 2, 2)
  plt.title("Max " + metric_used)
  plt.plot(index_max_distances_from_sample_ride_vehicle_1["long"], index_max_distances_from_sample_ride_vehicle_1["lat"])
  plt.plot(index_max_distances_from_sample_ride_vehicle_2["long"], index_max_distances_from_sample_ride_vehicle_2["lat"]) 
  plt.savefig("Min Max " + metric_used + ".png", bbox_inches = "tight")
  plt.close()
 
  only_NF = []
  only_NM = []
  only_I = [] 
  NF_and_NM = []
  NF_and_I = []
  NM_and_I = [] 
  for i in range(len(distances_from_sample_ride_vehicle)):
    if labels_from_sample_ride_vehicle[i][0] == "NF" and labels_from_sample_ride_vehicle[i][1] == "NF":
      only_NF.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "NM" and labels_from_sample_ride_vehicle[i][1] == "NM":
      only_NM.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "I" and labels_from_sample_ride_vehicle[i][1] == "I":
      only_I.append(distances_from_sample_ride_vehicle[i]) 
    if labels_from_sample_ride_vehicle[i][0] == "NF" and labels_from_sample_ride_vehicle[i][1] == "NM":
      NF_and_NM.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "NM" and labels_from_sample_ride_vehicle[i][1] == "NF":
      NF_and_NM.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "NF" and labels_from_sample_ride_vehicle[i][1] == "I":
      NF_and_I.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "I" and labels_from_sample_ride_vehicle[i][1] == "NF":
      NF_and_I.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "NM" and labels_from_sample_ride_vehicle[i][1] == "I":
      NM_and_I.append(distances_from_sample_ride_vehicle[i])
    if labels_from_sample_ride_vehicle[i][0] == "I" and labels_from_sample_ride_vehicle[i][1] == "NM":
      NM_and_I.append(distances_from_sample_ride_vehicle[i])
  
  plt.title("Dist " + metric_used)
  plt.hist(distances_from_sample_ride_vehicle)
  plt.hist(only_NF, label = "NF")
  plt.hist(only_NM, label = "NM")
  plt.hist(only_I, label = "I") 
  plt.hist(NF_and_NM, label = "NF NM")
  plt.hist(NF_and_I, label = "NF I")
  plt.hist(NM_and_I, label = "NM I") 
  plt.legend()
  plt.savefig("Dist " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  return distances_from_sample_ride_vehicle, labels_from_sample_ride_vehicle, trajs_from_sample_ride_vehicle, only_NF, only_NM, only_I, NF_and_NM, NF_and_I, NM_and_I 

#compare_all("custom")
#compare_all("dtw")
#compare_all("simpson")
#compare_all("trapz")
#compare_all("euclidean")

def get_all_surface(metric_used):
 
  distances_x = [] 
  distances_y = [] 
  labels_from_sample_ride_vehicle = [] 
  sl_from_sample_ride_vehicle = [] 
  max_x = 0
  index_max_x = 0 

  min_x = 100000
  index_min_x = 0 

  max_y = 0
  index_max_y = 0 

  min_y = 100000
  index_min_y = 0 

  for vehicle1 in all_possible_trajs[window_size].keys(): 
    for r1 in all_possible_trajs[window_size][vehicle1]: 
      for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
        t1 = all_possible_trajs[window_size][vehicle1][r1][x1]  
        #td_up_auc = abs(auc(t1["long"], t1["lat"]) - auc(sample_x, sample_y)) - isti kao trapz, x mora biti rastući ili padajući
        #print(td_up, td_up_trapz, td_up_simpson)#, td_up_auc)
        surface_x, surface_y = get_surf_xt_yt(t1["long"], t1["lat"], t1["time"], metric_used)
        distances_x.append(surface_x)
        distances_y.append(surface_y)
        labels_from_sample_ride_vehicle.append(trajectory_monotonous[window_size][vehicle1][r1][x1])
        sl_from_sample_ride_vehicle.append(trajectory_sl[window_size][vehicle1][r1][x1])

        if distances_x[-1] > max_x:
          index_max_x = t1
          max_x = distances_x[-1]
 
        if distances_x[-1] < min_x:
          index_min_x = t1
          min_x = distances_x[-1]

        if distances_y[-1] > max_y:
          index_max_y = t1
          max_y = distances_y[-1]
 
        if distances_y[-1] < min_y:
          index_min_y = t1
          min_y = distances_y[-1]

  plt.subplot(1, 2, 1)
  plt.title("Min x " + metric_used)
  plt.plot(index_min_x["long"], index_min_x["lat"])
  plt.subplot(1, 2, 2)
  plt.title("Max x " + metric_used)
  plt.plot(index_max_x["long"], index_max_x["lat"])
  plt.savefig("Min Max x " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  plt.subplot(1, 2, 1)
  plt.title("Min y " + metric_used)
  plt.plot(index_min_y["long"], index_min_y["lat"])
  plt.subplot(1, 2, 2)
  plt.title("Max y " + metric_used)
  plt.plot(index_max_y["long"], index_max_y["lat"])
  plt.savefig("Min Max y " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  only_T = {"x": [], "y": []}
  only_F = {"x": [], "y": []}
  only_NF = {"x": [], "y": []}
  only_NM = {"x": [], "y": []}
  only_I = {"x": [], "y": []} 
  only_NF_T = {"x": [], "y": []}
  only_NM_T = {"x": [], "y": []}
  only_I_T = {"x": [], "y": []} 
  only_NF_F = {"x": [], "y": []}
  only_NM_F = {"x": [], "y": []}
  only_I_F = {"x": [], "y": []} 
  for i in range(len(distances_x)):
    if sl_from_sample_ride_vehicle[i]:
        only_T["x"].append(distances_x[i])
        only_T["y"].append(distances_y[i])
    else:
        only_F["x"].append(distances_x[i])
        only_F["y"].append(distances_y[i]) 
 
    if labels_from_sample_ride_vehicle[i] == "NF":
      only_NF["x"].append(distances_x[i])
      only_NF["y"].append(distances_y[i])
      if sl_from_sample_ride_vehicle[i]:
        only_NF_T["x"].append(distances_x[i])
        only_NF_T["y"].append(distances_y[i])
      else:
        only_NF_F["x"].append(distances_x[i])
        only_NF_F["y"].append(distances_y[i]) 
           
    if labels_from_sample_ride_vehicle[i] == "NM":
      only_NM["x"].append(distances_x[i])
      only_NM["y"].append(distances_y[i])
      if sl_from_sample_ride_vehicle[i]:
        only_NM_T["x"].append(distances_x[i])
        only_NM_T["y"].append(distances_y[i])
      else:
        only_NM_F["x"].append(distances_x[i])
        only_NM_F["y"].append(distances_y[i]) 

    if labels_from_sample_ride_vehicle[i] == "I":
      only_I["x"].append(distances_x[i])
      only_I["y"].append(distances_y[i])
      if sl_from_sample_ride_vehicle[i]:
        only_I_T["x"].append(distances_x[i])
        only_I_T["y"].append(distances_y[i])
      else:
        only_I_F["x"].append(distances_x[i])
        only_I_F["y"].append(distances_y[i])
  
  plt.subplot(2, 2, 1)
  plt.title("Surf x NF " + metric_used)
  plt.hist(distances_x, label = "All")
  plt.hist(only_T["x"], label = "T")
  plt.hist(only_F["x"], label = "F")
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Surf x NF " + metric_used)
  plt.hist(only_NF["x"], label = "NF")
  plt.hist(only_NF_T["x"], label = "NF T")
  plt.hist(only_NF_F["x"], label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Surf x NM " + metric_used)
  plt.hist(only_NM["x"], label = "NM")
  plt.hist(only_NM_T["x"], label = "NM T")
  plt.hist(only_NM_F["x"], label = "NM F")
  plt.legend()
  plt.subplot(2, 2, 4)
  plt.title("Surf x I " + metric_used)
  plt.hist(only_I["x"], label = "I")
  plt.hist(only_I_T["x"], label = "I T")
  plt.hist(only_I_F["x"], label = "I F")
  plt.legend()
  plt.savefig("Surf x " + metric_used + ".png", bbox_inches = "tight")
  plt.close()

  plt.subplot(2, 2, 1)
  plt.title("Surf y NF " + metric_used)
  plt.hist(distances_y, label = "All")
  plt.hist(only_T["y"], label = "T")
  plt.hist(only_F["y"], label = "F")
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Surf y NF " + metric_used)
  plt.hist(only_NF["y"], label = "NF")
  plt.hist(only_NF_T["y"], label = "NF T")
  plt.hist(only_NF_F["y"], label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Surf y NM " + metric_used)
  plt.hist(only_NM["y"], label = "NM")
  plt.hist(only_NM_T["y"], label = "NM T")
  plt.hist(only_NM_F["y"], label = "NM F")
  plt.legend()
  plt.subplot(2, 2, 4)
  plt.title("Surf y I " + metric_used)
  plt.hist(only_I["y"], label = "I")
  plt.hist(only_I_T["y"], label = "I T")
  plt.hist(only_I_F["y"], label = "I F")
  plt.legend()
  plt.savefig("Surf y " + metric_used + ".png",  bbox_inches = "tight")
  plt.close()

  plt.subplot(2, 2, 1)
  plt.title("Scatter x y NF " + metric_used)
  plt.scatter(distances_x, distances_y, label = "All")
  plt.scatter(only_T["x"], only_T["y"], label = "T")
  plt.scatter(only_F["x"], only_F["y"], label = "F")
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Scatter x y NF " + metric_used)
  plt.scatter(only_NF["x"], only_NF["y"], label = "NF")
  plt.scatter(only_NF_T["x"], only_NF_T["y"], label = "NF T")
  plt.scatter(only_NF_F["x"], only_NF_F["y"], label = "NF F")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Scatter x y NM " + metric_used)
  plt.scatter(only_NM["x"], only_NM["y"], label = "NM")
  plt.scatter(only_NM_T["x"], only_NM_T["y"], label = "NM T")
  plt.scatter(only_NM_F["x"], only_NM_F["y"], label = "NM F")
  plt.legend()
  plt.subplot(2, 2, 4)
  plt.title("Scatter x y I " + metric_used)
  plt.scatter(only_I["x"], only_I["y"], label = "I")
  plt.scatter(only_I_T["x"], only_I_T["y"], label = "I T")
  plt.scatter(only_I_F["x"], only_I_F["y"], label = "I F")
  plt.legend()
  plt.savefig("Scatter x y " + metric_used + ".png",  bbox_inches = "tight")
  plt.close()
  
  return distances_x, distances_y, labels_from_sample_ride_vehicle, only_NF, only_NM, only_I

get_all_surface("trapz")
get_all_surface("simpson")

def get_all_fft():
  
  for vehicle1 in all_possible_trajs[window_size].keys(): 
    for r1 in all_possible_trajs[window_size][vehicle1]: 
      for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
        t1 = all_possible_trajs[window_size][vehicle1][r1][x1]  
        xt, yt, xn, yn, fftx, ffty = get_fft_xt_yt(t1["long"], t1["lat"], t1["time"])
      
        plt.plot(t1["time"], t1["long"]) 
        plt.plot(t1["time"], xn)
        plt.plot(t1["time"], fftx)
        plt.xlabel("seconds")
        plt.ylabel("x")
        plt.show() 

        decompose_fft(xn)

        plt.plot(t1["time"], t1["lat"]) 
        plt.plot(t1["time"], yn)
        plt.plot(t1["time"], ffty)
        plt.xlabel("seconds")
        plt.ylabel("y")
        plt.show() 

        decompose_fft(yn)
        
#get_all_fft()