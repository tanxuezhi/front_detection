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

slv_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv.nc'%(year)
slv_2_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv_2.nc'%(year)
slv_id = Dataset(slv_file, 'r')
slv_id.set_auto_mask(False)
my_lat = slv_id.variables['lat'][:]
my_lon = slv_id.variables['lon'][:]
my_slp = slv_id.variables['slp'][:]/100.
my_time = slv_id.variables['time'][:]
my_date = np.asarray([dt.datetime.fromordinal(int(i_time - 366.)) + dt.timedelta(hours=(i_time%1)*24.) for i_time in my_time])
my_lon, my_lat = np.meshgrid(my_lon, my_lat)
slv_id.close()


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

# creating the cdt grid 
lon, lat = np.meshgrid(in_lon, in_lat)

lev850 = np.where(in_lev == 850)[0][0]

print(' Completed!')

for t_step in range(1, in_time.shape[0]):

  # creating a datetime variable for the current time step
  date = dt.datetime(2007, 1, 1) + dt.timedelta(minutes=in_time[t_step])

  # getting catherinees fronts for the time step
  cath_wf, cath_cf, cath_slp, cath_lat, cath_lon = catherine.fronts_for_date(lat, lon, date.year, date.month, date.day, date.hour)
  
  llat = np.nanmin(cath_lat)
  ulat = np.nanmax(cath_lat)
  llon = np.nanmin(cath_lon)
  ulon = np.nanmax(cath_lon)

  # getting the different slp values for MERRA2
  my_t_slp = np.squeeze(my_slp[(my_date == date), :, :])
  slp = in_slp[t_step, :, :]/100.

  # plt.figure(figsize=(3,9))
  # plt.subplot(311)
  # m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  # m.contourf(lon, lat, slp, levels=np.arange(960, 1100, 10), cmap='jet')
  # m.drawcoastlines()
  # m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
  # m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
  # m.colorbar()
  # plt.title('INST6_3d_ANA_NP')
  # plt.subplot(312)
  # m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  # m.contourf(my_lon, my_lat, my_t_slp, levels=np.arange(960, 1100, 10), cmap='jet')
  # m.colorbar()
  # m.drawcoastlines()
  # m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
  # m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
  # plt.title('MY 6H averages SLP')
  # plt.subplot(313)
  # m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  # m.contourf(cath_lon, cath_lat, cath_slp, levels=np.arange(960, 1100, 10), cmap='jet')
  # m.colorbar()
  # m.drawcoastlines()
  # m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
  # m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
  # plt.title('CATH SLP')
  # plt.tight_layout()
  # plt.savefig('./images/slp_compare.png', dpi=300.)
  # plt.close('all')

  # extracting the current and previous time step U & V wind speeds for the fronts
  # have to smooth the input data, catherine smooths it 10 times, so do I
  # weighting the center point 4x as heavier 
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
  pres = np.repeat(in_lev[:, np.newaxis], H.shape[1], axis=-1) # creating the pressure level into 3d array
  pres = np.repeat(pres[:, :, np.newaxis], H.shape[2], axis=-1) # creating the pressure level into 3d array 
  pres = np.ma.masked_array(pres, mask=~idx, fill_value=np.nan) # masking out pressure values using minimum 1km mask
  p1km = np.nanmin(pres, axis=0) # getting the pressure at 1km
  theta1km = fd.theta_from_temp_pres(t1km, p1km) # computing the theta value at 1km
  theta1km = fd.smooth_grid(theta1km, iter=10, center_weight=4) # smoothing out the theta value
 
  # computing the simmonds fronts
  f_sim = fd.simmonds_et_al_2012(lat, lon, prev_u850, prev_v850, u850, v850) 

  # computing the hewson fronts using 1km temperature values, and U & V wind speeds at 850
  # f_hew = fd.hewson_1998(lat, lon, theta1km, u850, v850)

  f_hew = fd.hewson_1998(lat, lon, theta850, u850, v850)
  # zc_6, zc_7 = fd.hewson_1998(lat, lon, theta850, u850, v850)
  
  wf_hew = f_hew['wf']
  cf_hew = f_hew['cf']
  cf_sim = f_sim['cf']

  wf = np.copy(wf_hew)
  # cf = np.double((cf_hew + cf_sim) > 0)
  cf = np.copy(cf_sim)
 
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

  plt.figure(figsize=(12,12))
  fronts = wf*10 + cf*-10
  fronts = cf*-10
  fronts[~((fronts == 10) | (fronts == -10))] = np.nan
  plt.subplot(2,1,1)
  m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  csf = plt.contourf(lon, lat, theta850)
  cs = plt.contour(lon, lat, slp, lw=0.5, ls='--', colors='k')
  plt.clabel(cs, inline=1., fontsize=10., fmt='%.0f')
  pc = m.pcolormesh(lon, lat, fronts, cmap='bwr')
  m.colorbar(csf)
  m.drawcoastlines(linewidth=0.2)
  plt.axhline(y=0., linewidth=1.0, linestyle='--')
  plt.title('My Fronts')

  plt.subplot(2,1,2)
  fronts = cath_wf*10 + cath_cf*-10
  fronts = cath_cf*-10
  fronts[~((fronts == 10) | (fronts == -10))] = np.nan
  m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  # csf = plt.contourf(cath_lon, cath_lat, cath_slp)
  csf = plt.contourf(lon, lat, theta850)
  cs = plt.contour(lon, lat, slp, lw=0.5, ls='--', colors='k')
  plt.clabel(cs, inline=1., fontsize=10., fmt='%.0f')
  pc = m.pcolormesh(lon, lat, fronts, cmap='bwr')
  m.colorbar(csf)
  m.drawcoastlines(linewidth=0.2)
  plt.axhline(y=0., linewidth=1.0, linestyle='--')
  plt.title('Catherine Fronts')

  plt.savefig('./images/test.png', dpi=300)
  plt.show()
  
  break

  # plt.savefig('./images/test.png', dpi=300.)

# slv2_id.close()
ncid.close()

'''
### temp codes -- delete later
# code to test slp data
plt.close('all')
llat = np.nanmin(cath_lat)
ulat = np.nanmax(cath_lat)
llon = np.nanmin(cath_lon)
ulon = np.nanmax(cath_lon)

my_ind = (my_date == date)
my_t_slp = np.squeeze(my_slp[my_ind, :, :])
slp = in_slp[t_step, :, :]/100.

plt.figure(figsize=(3,9))
plt.subplot(311)
m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
m.contourf(lon, lat, slp, levels=np.arange(960, 1100, 10), cmap='jet')
m.drawcoastlines()
m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
m.colorbar()
plt.title('INST6_3d_ANA_NP')

plt.subplot(312)
m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
m.contourf(my_lon, my_lat, my_t_slp, levels=np.arange(960, 1100, 10), cmap='jet')
m.colorbar()
m.drawcoastlines()
m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
plt.title('MY 6H averages SLP')

plt.subplot(313)
m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
m.contourf(cath_lon, cath_lat, cath_slp, levels=np.arange(960, 1100, 10), cmap='jet')
m.colorbar()
m.drawcoastlines()
m.drawparallels(np.arange(-90, 90, 30), labels=[False, True, False, False])
m.drawmeridians(np.arange(-180, 180, 30), labels=[False, False, False, True])
plt.title('CATH SLP')

plt.tight_layout()
plt.savefig('./images/slp_compare.png', dpi=300.)

ncid.close()
'''
