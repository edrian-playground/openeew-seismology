"""
This file sets parameters used in real-time OpenEEW algorithm
"""

# MQTT
MQTT = "IBM"  # local, custom, or IBM

# TRAVEL TIME GRID AND CALCULATION
lat_width = 20  # latitude grid width
lon_width = 20  # longitude grid width
step = 0.01  # step in degrees
eq_depth = 20  # earthquake depth
vel_model = "iasp91"  # velocity model from obspy list
tt_path = "./obj/travel_times"  # relative path to the travel_time folder
buffer_len = 15  # buffer_len*samp_rate must be longer than array_samp

# DETECTION
det_type = "stalta"  # 'stalta' or 'ml' for machine learning
detection_model_name = "detection_model.model"  # name of the ml model
STA_len = 1  # STA length in samples
LTA_len = 8  # LTA length in samples
array_samp = 352  # must be >= STA_len+LTA_len for 'stalta', or 300 for 'ml'
STALTA_thresh = 3  # threshold for STA/LTA
max_std = 0.08  # detect only if the STD of LTA window is under this value
no_det_win = 60  # window without new detections after a detection
vert_chan = "y"  # which channel is oriented in the vertical direction
sleep_time = 1  # the detection algorithm pauses for this time after each loop
plot_detection = False  # do you want to plot and save detections?
plot_event = False  # do you want to plot and save events?

# DEVICE DATABASE
sleep_time_devices = 10  # the update device table after this time
db_name = "openeew-devices"
device_local_path = "./data/devices/device_locations.json"

# LOCATION AND MAGNITUDE REGRESSION PARAMS
tsl_max = 60  # save/discard event after this many seconds without a new detection
assoc_win = 2  # window for associated phases
ndef_min = 4  # minimum number of station detections defining an event
sigma_type = "const"  # either 'const' sigma or 'linear' function
sigma_const = 3  # overall time error (travel time + pick + cloud_time)
nya_weight = 1  # how much to weight not-yet-arrived information
nya_nos = 1  # use not-yet-arrived information for this number of seconds after the first arrival
prior_type = (
    "constant"  # 'constant' or 'gutenberg' if you like to start with GR distribution
)
mc = 3  # magnitude of completeness for GR distribution
b_value = 1  # b-value for GR distribution
sleep_time = 1  # the event algorithm is going to pause for this time after each loop

# a, b, c, std params in M = a*pd + b, c is distance normalization, std is pd scatter
mag1 = (1.67, 5.68, 1, 0.85)
mag2 = (1.56, 5.47, 1, 0.74)
mag3 = (1.44, 5.35, 1, 0.66)
mag4 = (1.41, 5.32, 1, 0.59)
mag5 = (1.41, 5.29, 1, 0.57)
mag6 = (1.35, 5.22, 1, 0.51)
mag7 = (1.45, 5.24, 1, 0.57)
mag8 = (1.39, 5.21, 1, 0.52)
mag9 = (1.32, 5.19, 1, 0.47)


params = {
    "MQTT": MQTT,
    "lat_width": lat_width,
    "lon_width": lon_width,
    "step": step,
    "vel_model": vel_model,
    "eq_depth": eq_depth,
    "tt_path": tt_path,
    "det_type": det_type,
    "STA_len": STA_len,
    "LTA_len": LTA_len,
    "STALTA_thresh": STALTA_thresh,
    "max_std": max_std,
    "no_det_win": no_det_win,
    "vert_chan": vert_chan,
    "array_samp": array_samp,
    "detection_model_name": detection_model_name,
    "buffer_len": buffer_len,
    "sleep_time": sleep_time,
    "plot_detection": plot_detection,
    "plot_event": plot_event,
    "sleep_time_devices": sleep_time_devices,
    "db_name": db_name,
    "device_local_path": device_local_path,
    "mag1": mag1,
    "mag2": mag2,
    "mag3": mag3,
    "mag4": mag4,
    "mag5": mag5,
    "mag6": mag6,
    "mag7": mag7,
    "mag8": mag8,
    "mag9": mag9,
    "tsl_max": tsl_max,
    "ndef_min": ndef_min,
    "sigma_const": sigma_const,
    "sigma_type": sigma_type,
    "nya_weight": nya_weight,
    "nya_nos": nya_nos,
    "prior_type": prior_type,
    "mc": mc,
    "b_value": b_value,
    "assoc_win": assoc_win,
    "eq_depth": eq_depth,
    "sleep_time": sleep_time,
}
