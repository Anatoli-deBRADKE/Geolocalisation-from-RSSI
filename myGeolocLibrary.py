import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from geopy.distance import vincenty


# Calculate the grate circle distance between two points X and Y on the earth (specified in decimal degrees)
def haversine(lon_x, lat_x, lon_y, lat_y):

    # convert decimal degrees to radians
    lon_x, lat_x, lon_y, lat_y = map(radians, [lon_x, lat_x, lon_y, lat_y])

    # haversine formula
    a = sin((lat_y - lat_x)/2) ** 2 + cos(lat_x) * cos(lat_y) * sin((lon_y - lon_x)/2) ** 2
    c = 2 * asin(sqrt(a))

    # Radius of earth
    r = 6372.8

    # Return result
    return c * r



# Compute distance between an base station and a message
def compute_dist_msg_bs(row):

    # Retrieve latitude and longitude of message and base station
    bs_lat = row["bs_lat"]
    bs_lng = row["bs_lng"]
    msg_lat = row["mess_lat"]
    msg_lng = row["mess_lng"]

    # Compute haversine
    return haversine(bs_lng, bs_lat, msg_lng, msg_lat)



# Feature Matrix construction
def feat_mat_const(df, listOfBs):

    # Group data by message ID
    df_mess_bs_group = df.groupby(['messid'], as_index=False)

    # Number of message ID
    nb_mess = len(np.unique(df['messid']))

    # Features name list
    columns_list = listOfBs.tolist() + [x + max(listOfBs) for x in listOfBs.tolist()]
    
    # Initialize feature dataframe
    df_feat = pd.DataFrame(np.zeros((nb_mess, 2 * len(listOfBs))), columns = columns_list)

    # Calculate features for each message ID
    idx = 0
    id_list = [0] * nb_mess
    for key, elmt in df_mess_bs_group:

        # get all base station for one message ID
        df_mess_bs_group.get_group(key)

        # Exponential rssi
        df_feat.loc[idx,df_mess_bs_group.get_group(key)['bsid']] = np.array(list(np.exp(df_mess_bs_group.get_group(key)['rssi'])))

        # Dummies
        df_feat.loc[idx,df_mess_bs_group.get_group(key)['bsid'] + max(listOfBs)] = 1
        
        # Get message id list
        id_list[idx] = key

        # Next message
        idx = idx + 1

    # Return results
    return df_feat, id_list


# Ground truth construction
def ground_truth_const(df, pos):

    # Combine dataframe and related message position
    df_mess_pos = df.copy()
    df_mess_pos[['lat', 'lng']] = pos
    ground_truth_lat = np.array(df_mess_pos.groupby(['messid']).mean()['lat'])
    ground_truth_lng = np.array(df_mess_pos.groupby(['messid']).mean()['lng'])

    # Return results
    return ground_truth_lat, ground_truth_lng


# Train regressor and make prediction in the train set
def regressor_and_predict(regressor_lat, regressor_lng, df_feat, ground_truth_lat, ground_truth_lng, df_test):

    # Cast feature dataframe to array matrix
    X_train = np.array(df_feat);

    # Fit model for latitude
    regressor_lat.fit(X_train, ground_truth_lat)

    # Predict latitude
    y_pred_lat = regressor_lat.predict(df_test)

    # Fit model for longitude
    regressor_lng.fit(X_train, ground_truth_lng)

    # Predict longitude
    y_pred_lng = regressor_lng.predict(df_test)

    # Return results
    return y_pred_lat, y_pred_lng


# Cross Validation and hyper parameter tunning
def cv_best_param(regressor, param_grid, df_feat, ground_truth_lat, ground_truth_lng, df_test, cv):

    # Cast feature dataframe to array matrix
    X_train = np.array(df_feat);

    # Cross Validation
    cv = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=cv, n_jobs=-1)

    # Fit model for latitude
    cv.fit(X_train, ground_truth_lat)

    # Predict latitude
    best_param_lat = cv.best_estimator_

    # Fit model for longitude
    cv.fit(X_train, ground_truth_lng)

    # Predict longitude
    best_param_lng = cv.best_estimator_

    # Return results
    return best_param_lat, best_param_lng


# Compute vincenty
def vincenty_vec(vec_coord):
    vin_vec_dist = np.zeros(vec_coord.shape[0])
    if vec_coord.shape[1] !=  4:
        print('ERROR: Bad number of columns (shall be = 4)')
    else:
        vin_vec_dist = [vincenty(vec_coord[m,0:2],vec_coord[m,2:]).meters for m in range(vec_coord.shape[0])]
    return vin_vec_dist



# Evaluate distance error for each predicted point
def Eval_geoloc(y_train_lat , y_train_lng, y_pred_lat, y_pred_lng):
    vec_coord = np.array([y_train_lat , y_train_lng, y_pred_lat, y_pred_lng])
    err_vec = vincenty_vec(np.transpose(vec_coord))

    return err_vec

#################################################################################
#                   Function to enable zoom on plot
#################################################################################

def zoom_factory(ax,base_scale = 1.2):
    def zoom_fun(event):
        
        # get the current x and y limits
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        # Get distance from the cursor to the edge of the figure frame
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata
        
        if (event.button) == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif (event.button) == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print (event.button)
        
        ax.set_xlim([xdata - x_left*scale_factor,
                    xdata + x_right*scale_factor])
        ax.set_ylim([ydata - y_top*scale_factor,
                    ydata + y_bottom*scale_factor])
        
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

#################################################################################
#                   Function to save positions to csv format
#################################################################################

def coords_to_csv(lats, lngs, types=None):
    if types is not None :
        pd.DataFrame(np.array([types, lats, lngs]).T).to_csv("data.csv", header=None, index=False)
    else :
        pd.DataFrame(np.array([lats, lngs]).T).to_csv("data.csv", header=None, index=False)
    return

from datetime import datetime

#################################################################################
#                   Function to convert unix time to timestamp
#################################################################################

def ux_to_timestamp(ts):
    timestamp = None
    try:
        timestamp = datetime.utcfromtimestamp(ts//1000).strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
        
    return timestamp