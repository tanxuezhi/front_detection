import scipy.io as sio
import numpy as np 
from netCDF4 import Dataset

import matplotlib.pyplot as plt

year_list = [2014, 2018]
year = 2015

in_file = '/mnt/drive1/processed_data/tracks/merra2_tracks/ERAI_%d_cyc.mat'%(year)

data = sio.loadmat(in_file)['cyc']

lat = []
lon = []
for track_lat, track_lon in zip(np.squeeze(data['fulllat']), np.squeeze(data['fulllon'])):
  lat.extend(track_lat)
  lon.extend(track_lon)

lat = np.asarray(lat)
lon = np.asarray(lon)

# nc = Dataset('./data/tracked_fronts/centers_%d.nc'%(year), 'w')
# nc.close()
