#!/usr/bin/env python
import sys
sys.path.append('..')
import numpy as np 
import front_detection as fd
import scipy.io as sio
from scipy.ndimage import label, generate_binary_structure
from scipy import stats
import os
import glob
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime as dt
import plotter

year = 2007
model_name = 'merra2'
hemis = 'NH'

folder_format = '/localdrive/drive10/jj/datacycs/out_nc/{0}/{1}/{2}/' 

model_folder = '/mnt/drive5/merra2/six_hrly/'

slv_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv.nc'%(year)
slv_2_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv_2.nc'%(year)

slv_id = Dataset(slv_file, 'r')
slp = slv_id.variables['slp'][:]
slv_id.close()

# loading in merra2 inst6_3d_ana_Np data
ncid = Dataset('/localdrive/drive10/merra2/inst6_3d_ana_Np/MERRA2_300.inst6_3d_ana_Np.20070101.nc4', 'r')
ncid.set_auto_mask(False)
in_lon = ncid.variables['lon'][:]
in_lat = ncid.variables['lat'][:]
lev = ncid.variables['lev'][:]
time = np.asarray(ncid.variables['time'][:], dtype=float)
slp = ncid.variables['SLP'][:]

T = ncid.variables['T']
U = ncid.variables['U']
V = ncid.variables['V']
geoH = ncid.variables['H']

# creating the cdt grid 
lon, lat = np.meshgrid(in_lon, in_lat)

lev850 = np.where(lev == 850)[0][0]

for t_step in range(1, time.shape[0]):

  # creating a datetime variable for the current time step
  date = dt.datetime(2007, 1, 1) + dt.timedelta(minutes=time[t_step])

  # getting catherinees fronts for the time step
  cath_wf, cath_cf = fd.catherine_fronts_for_date(lat, lon, date.year, date.month, date.day, date.hour)

  plt.figure()
  m = Basemap(projection='cyl', urcrnrlat=90, llcrnrlat=0, urcrnrlon=0, llcrnrlon=-180)
  pc = m.pcolormesh(lon, lat, cath_wf*10 + cath_cf*-10, cmap='bwr')
  m.contour(lon, lat, slp[t_step, :, :], lw=1.0, levels=[960, 980, 1000])
  m.drawcoastlines(linewidth=0.2)
  m.colorbar(pc)
  plt.title('Catherine Fronts [Hewson 1km]')
  plt.savefig('./images/cath_hew_1km.png', dpi=300)
  plt.show()

  break

 
  # extracting the current and previous time step U & V wind speeds for the fronts
  prev_u850 = fd.smooth_grid(U[t_step-1, lev850, :, :], iter=10, center_weight=4)
  u850 = fd.smooth_grid(U[t_step, lev850, :, :], iter=10, center_weight=4) 

  prev_v850 = fd.smooth_grid(V[t_step-1, lev850, :, :], iter=10, center_weight=4)
  v850 = fd.smooth_grid(V[t_step, lev850, :, :], iter=10, center_weight=4) 
 
  # getting the temperature at 850 hPa
  t850 = T[t_step, lev850, :, :]
  t850[t850 > 1000] = np.nan
  theta850 = fd.theta_from_temp_pres(t850, 850)
  theta850 = fd.smooth_grid(theta850, iter=10, center_weight=4) 

  # getting the 1km values of temperature
  # the code below is a work around to speed up the code, isntead of running a nest for loop

  # getting the height values from MERRA2
  H = geoH[t_step, :, :, :]/9.8
  H1km_diff = np.abs(H - 1000.) # getting the difference between Height and 1km, to get the min value
  min_val = np.broadcast_to(np.nanmin(H1km_diff, axis=0), (H.shape[0], H.shape[1], H.shape[2])) # getting a min_val array to mask out the main array to find the closest minimum value
  idx = (H1km_diff == min_val) # getting the index mask of all the minimum values 
  T_3d = np.ma.masked_array(T[t_step, :, :, :], mask=~idx, fill_value=np.nan) # creating a temperature 3d array
  t1km = np.nanmin(T_3d.filled(),axis=0)  # getting the 1km value by finding the minimum value
  pres = np.repeat(lev[:, np.newaxis], H.shape[1], axis=-1) # creating the pressure level into 3d array
  pres = np.repeat(pres[:, :, np.newaxis], H.shape[2], axis=-1) # creating the pressure level into 3d array 
  pres = np.ma.masked_array(pres, mask=~idx, fill_value=np.nan) # masking out pressure values using minimum 1km mask
  p1km = np.nanmin(pres, axis=0) # getting the pressure at 1km
  theta1km = fd.theta_from_temp_pres(t1km, p1km) # computing the theta value at 1km
  theta1km = fd.smooth_grid(theta1km, iter=10, center_weight=4) # smoothing out the theta value
 
  # computing the simmonds fronts
  cf_sim = fd.simmonds_et_al_2012(lat, lon, prev_u850, prev_v850, u850, v850) 

  # computing the hewson fronts using 1km temperature values, and U & V wind speeds at 850
  f_hew = fd.hewson_1998(lat, lon, theta1km, u850, v850)

  # f_hew = fd.hewson_1998(lat, lon, theta850, u850, v850)
  # zc_6, zc_7 = fd.hewson_1998(lat, lon, theta850, u850, v850)
  
  wf_hew = f_hew['wf']
  cf_hew = f_hew['cf']
  cf_sim = f_sim['cf']

  wf = np.copy(wf_hew)
  cf = np.double((cf_hew + cf_sim) > 0)
  # cf = np.copy(cf_sim)
 
  ## Cleaning up the fronts
  s = generate_binary_structure(2,2)
  w_label, w_num = label(wf, structure=s)
  c_label, c_num = label(cf, structure=s)

  # keeping only clusters with 3 or more 
  for i_w in range(1, w_num+1):
    ind = np.argwhere(w_label == i_w)
    if (len(ind) < 3):
      wf[w_label == i_w] = 0.

  # cleaning up the cold fronts and picking only the eastern most point
  # cf_old = np.copy(cf)
  for i_c in range(1, c_num+1):
    x_ind, y_ind = np.where(c_label == i_c)

    if (len(x_ind) < 3):
      cf[c_label == i_c] = 0.
      # cf_old[c_label == i_c] = 0.
      continue

    # quick scatched up way to keep only eastern most points
    # optimize this later
    for uni_x in set(x_ind):
      y_for_uni_x = y_ind[(x_ind == uni_x)]
      remove_y = y_for_uni_x[y_for_uni_x != np.nanmax(y_for_uni_x)]
      if (remove_y.size > 0):
        for y in remove_y: 
          cf[uni_x, y] = 0.

  plt.figure()
  plt.subplot(2,1,1)
  m = Basemap(projection='cyl', urcrnrlat=90, llcrnrlat=0, urcrnrlon=0, llcrnrlon=-180)
  m.contour(lon, lat, slp[t_step, :, :], lw=1.0, levels=[960, 980, 1000])
  m.pcolormesh(lon, lat, wf*10 + cf*-10, cmap='bwr')
  m.drawcoastlines(linewidth=0.2)
  m.colorbar()
  plt.title('My Fronts')

  plt.subplot(2,1,2)
  m = Basemap(projection='cyl', urcrnrlat=90, llcrnrlat=0, urcrnrlon=0, llcrnrlon=-180)
  m.contour(lon, lat, slp[t_step, :, :], lw=1.0, levels=[960, 980, 1000])
  m.pcolormesh(lon, lat, cath_wf*10 + cath_cf*-10, cmap='bwr')
  m.drawcoastlines(linewidth=0.2)
  m.colorbar()
  plt.title('Catherine Fronts')

  plt.savefig('./images/test.png', dpi=300)

  break

  # plt.savefig('./images/test.png', dpi=300.)

# slv2_id.close()
ncid.close()
