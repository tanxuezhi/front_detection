#!/usr/bin/env python
import numpy as np 
import front_detection as fd
from front_detection import catherine
from scipy.ndimage import label, generate_binary_structure
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

print('Debug: Reading in data ...', end='')

# loading in merra2 inst6_3d_ana_Np data
ncid = Dataset('/localdrive/drive10/merra2/inst6_3d_ana_Np/MERRA2_300.inst6_3d_ana_Np.20070101.nc4', 'r')
ncid.set_auto_mask(False)
in_lon = ncid.variables['lon'][:]
in_lat = ncid.variables['lat'][:]
in_lev = ncid.variables['lev'][:]
in_time = np.asarray(ncid.variables['time'][:], dtype=float)

in_slp = ncid.variables['SLP']
T = ncid.variables['T']
U = ncid.variables['U']
V = ncid.variables['V']
geoH = ncid.variables['H']
PS = ncid.variables['PS']

# creating the cdt grid 
lon, lat = np.meshgrid(in_lon, in_lat)

lev850 = np.where(in_lev == 850)[0][0]

print(' Completed!')

for t_step in range(1, in_time.shape[0]):

  # creating a datetime variable for the current time step
  date = dt.datetime(2007, 1, 1) + dt.timedelta(minutes=in_time[t_step])

  # getting SLP values for MERRA2
  slp = in_slp[t_step, :, :]/100.
  
  # getting Wind and temperature at 850 hPa for MERRA2
  prev_u850 = U[t_step-1, lev850, :, :]
  prev_u850[prev_u850 == U._FillValue] = np.nan
  u850 = U[t_step, lev850, :, :]
  u850[u850 == U._FillValue] = np.nan

  prev_v850 = V[t_step-1, lev850, :, :]
  prev_v850[prev_v850 == V._FillValue] = np.nan
  v850 = V[t_step, lev850, :, :]
  v850[v850 == V._FillValue] = np.nan
 
  # getting the temperature at 850 hPa
  t850 = T[t_step, lev850, :, :]
  t850[t850 == T._FillValue] = np.nan
  theta850 = fd.theta_from_temp_pres(t850, 850)

  # getting the 1km values of temperature
  # the code below is a work around to speed up the code, isntead of running a nest for loop

  # getting the height values from MERRA2
  H = geoH[t_step, :, :, :]
  H[H == geoH._FillValue] = np.nan
  H850 = geoH[t_step, lev850, :, :]
  H850[H850 == geoH._FillValue] = np.nan
  # H = H / 9.8

  # getting the surface pressure in hPa 
  surface_pres = PS[t_step, :, :]/100.
  surface_pres[surface_pres == PS._FillValue] = np.nan

  pres_3d = np.repeat(in_lev[:, np.newaxis], H.shape[1], axis=-1) # creating the pressure level into 3d array
  pres_3d = np.repeat(pres_3d[:, :, np.newaxis], H.shape[2], axis=-1) # creating the pressure level into 3d array 

  ps_3d = np.repeat(surface_pres[np.newaxis,:,:], H.shape[0], axis=0)

  # getting the surface height using the surface pressure and geo-potential height 
  pres_diff = np.abs(pres_3d - ps_3d)
  pres_diff_min_val = np.nanmin(pres_diff, axis=0) 
  pres_diff_min_val3d = np.repeat(pres_diff_min_val[np.newaxis, :, :], H.shape[0], axis=0)
  surface_H_ind = (pres_diff == pres_diff_min_val3d)
  ps_height = np.ma.masked_array(np.copy(H), mask=~surface_H_ind, fill_value=np.nan)  
  ps_height = np.nanmin(ps_height.filled(), axis=0) # surface height in km 

  # 1km height above surface
  h1km = ps_height + 1000. 
  h1km_3d = np.repeat(h1km[np.newaxis, :, :], H.shape[0], axis=0); 

  # difference between geopotential height and 1km height
  h1km_diff = np.abs(H - h1km_3d) 
  h1km_diff_min_val = np.nanmin(h1km_diff, axis=0) 
  h1km_diff_min_val_3d = np.repeat(h1km_diff_min_val[np.newaxis, :, :], H.shape[0], axis=0)
  h1km_ind = (h1km_diff == h1km_diff_min_val_3d)

  T_3d = np.ma.masked_array(T[t_step, :, :, :], mask=~h1km_ind, fill_value=np.nan) # creating a temperature 3d array
  t1km = np.nanmin(T_3d.filled(),axis=0)  # getting the 1km value by finding the minimum value
  t1km[t1km == T._FillValue] = np.nan
  
  U_3d = np.ma.masked_array(U[t_step, :, :, :], mask=~h1km_ind, fill_value=np.nan) # creating a temperature 3d array
  u1km = np.nanmin(U_3d.filled(),axis=0)  # getting the 1km value by finding the minimum value
  u1km[u1km == U._FillValue] = np.nan
  
  V_3d = np.ma.masked_array(V[t_step, :, :, :], mask=~h1km_ind, fill_value=np.nan) # creating a temperature 3d array
  v1km = np.nanmin(V_3d.filled(),axis=0)  # getting the 1km value by finding the minimum value
  v1km[v1km == V._FillValue] = np.nan

  pres = np.ma.masked_array(pres_3d, mask=~h1km_ind, fill_value=np.nan) # masking out pressure values using minimum 1km mask
  p1km = np.nanmin(pres.filled(), axis=0) # getting the pressure at 1km

  # computing the theta value at 1km
  theta1km = fd.theta_from_temp_pres(t1km, p1km) 

  # smoothing out the read in arrays
  iter_smooth = 10
  center_weight = 4.

  theta850 = fd.smooth_grid(theta850, iter=iter_smooth, center_weight=center_weight) 
  theta1km = fd.smooth_grid(theta1km, iter=iter_smooth, center_weight=center_weight) 

  u1km = fd.smooth_grid(u1km, iter=iter_smooth, center_weight=center_weight) 
  v1km = fd.smooth_grid(v1km, iter=iter_smooth, center_weight=center_weight) 

  prev_u850 = fd.smooth_grid(prev_u850, iter=iter_smooth, center_weight=center_weight) 
  u850 = fd.smooth_grid(u850, iter=iter_smooth, center_weight=center_weight) 
  prev_v850 = fd.smooth_grid(prev_v850, iter=iter_smooth, center_weight=center_weight) 
  v850 = fd.smooth_grid(v850, iter=iter_smooth, center_weight=center_weight) 

  # computing the simmonds fronts
  f_sim = fd.simmonds_et_al_2012(lat, lon, prev_u850, prev_v850, u850, v850) 

  # computing the hewson fronts using 1km temperature values, and geostrophic U & V winds at 850 hPa
  f_hew, var = fd.hewson_1998(lat, lon, theta1km, H850)
  
  wf_hew = f_hew['wf']
  cf_hew = f_hew['cf']
  cf_sim = f_sim['cf']

  wf_hew[np.isnan(wf_hew)] = 0
  cf_hew[np.isnan(cf_hew)] = 0
  cf_sim[np.isnan(cf_sim)] = 0

  wf = np.copy(wf_hew)
  cf = np.double((cf_hew + cf_sim) > 0)
  # cf = np.copy(cf_hew)
 
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

  llat = 0
  ulat = 90
  llon = -180
  ulon = 0 

  plt.figure(figsize=(12,8))
  fronts = wf*10 + cf*-10
  # fronts = cf*-10
  fronts[~((fronts == 10) | (fronts == -10))] = np.nan
  m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  csf = plt.contourf(lon, lat, var)
  # csf = plt.contourf(lon, lat, slp)
  cs = plt.contour(lon, lat, slp, lw=0.5, ls='--', colors='k', levels=np.arange(980, 1100, 5))
  plt.clabel(cs, inline=1., fontsize=10., fmt='%.0f')
  pc = m.pcolormesh(lon, lat, fronts, cmap='bwr')
  m.colorbar(csf)
  m.drawcoastlines(linewidth=0.2)
  plt.axhline(y=0., linewidth=1.0, linestyle='--')
  plt.title('My Fronts')

  plt.show()
  
  break

ncid.close()

