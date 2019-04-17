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

# slv_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv.nc'%(year)
# slv_2_file = '/mnt/drive5/merra2/six_hrly/MERRA_%d_slv_2.nc'%(year)
# slv_id = Dataset(slv_file, 'r')
# slv_id.set_auto_mask(False)
# my_lat = slv_id.variables['lat'][:]
# my_lon = slv_id.variables['lon'][:]
# my_slp = slv_id.variables['slp'][:]/100.
# my_time = slv_id.variables['time'][:]
# my_date = np.asarray([dt.datetime.fromordinal(int(i_time - 366.)) + dt.timedelta(hours=(i_time%1)*24.) for i_time in my_time])
# my_lon, my_lat = np.meshgrid(my_lon, my_lat)
# slv_id.close()


cid = Dataset('./data/input.cdf')
cid.set_auto_mask(False)
c_lon = cid.variables['longitude'][:]
c_lat = cid.variables['latitude'][:]
c_slp = cid.variables['SLP'][:]
c_t1km = cid.variables['temp1km'][:]
c_theta1km = cid.variables['theta1km'][:]
c_pres1km = cid.variables['press1km'][:]
c_vgx = cid.variables['vgx'][:]
c_vgy = cid.variables['vgy'][:]
cid.close()

cid = Dataset('./data/output.cdf')
cid.set_auto_mask(False)
c_mux = cid.variables['mux'][:]
c_muy = cid.variables['muy'][:]
c_m1 = cid.variables['m1'][:]
c_m2 = cid.variables['m2'][:]
c_betamean = cid.variables['betamean'][:]
c_dmean = cid.variables['dmean'][:]
c_dsdiv = cid.variables['dsdiv'][:]
c_flon = cid.variables['frontlong'][:]
c_flat = cid.variables['frontlat'][:]
c_ftype = cid.variables['fronttype'][:]
c_gta = cid.variables['geothermadv'][:]
cid.close()

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

  # getting catherinees fronts for the time step
  cath_wf, cath_cf, cath_slp, cath_lat, cath_lon, cf_lat, cf_lon, wf_lat, wf_lon = catherine.fronts_for_date(lat, lon, date.year, date.month, date.day, date.hour)
  
  # getting the different slp values for MERRA2
  # my_t_slp = np.squeeze(my_slp[(my_date == date), :, :])
  slp = in_slp[t_step, :, :]/100.

  # extracting the current and previous time step U & V wind speeds for the fronts
  # have to smooth the input data, catherine smooths it 10 times, so do I
  # weighting the center point 4x as heavier 
  
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

  H850 = fd.smooth_grid(H850, iter=iter_smooth, center_weight=center_weight) 

  ############# JJ TEST START ################
  gx, gy = fd.geo_gradient(lat, lon, theta1km)
  gNorm = fd.norm(gx, gy)
  mux, muy = fd.geo_gradient(lat, lon, gNorm)

  mux[np.abs(lat) > 70] = np.nan
  c_mux[np.abs(lat) > 70] = np.nan
  muy[np.abs(lat) > 70] = np.nan
  c_muy[np.abs(lat) > 70] = np.nan
  mux[np.abs(lon) > 170] = np.nan
  c_mux[np.abs(lon) > 170] = np.nan
  muy[np.abs(lon) > 170] = np.nan
  c_muy[np.abs(lon) > 170] = np.nan

  abs_mu = fd.norm(mux, muy)
  grad_abs_mux, grad_abs_muy = fd.geo_gradient(lat, lon, abs_mu)

  product = (mux * gx + muy*gy)
  product_smooth = fd.smooth_grid(product, iter=1, center_weight=1)
  m1 = -1*product_smooth/gNorm
  m2 = gNorm + (1/np.sqrt(2)) * 100 * 1000 * abs_mu

  k1 = 0.33 * 1e-10
  k2 = 1.49 * 1e-5

  m1_mask = m1 > k1
  m2_mask = m2 > k2

  c_m2[np.abs(lat) > 70] = np.nan
  c_m1[np.abs(lat) > 70] = np.nan
  m2[np.abs(lat) > 70] = np.nan
  m1[np.abs(lat) > 70] = np.nan
  c_m2[np.abs(lon) > 170] = np.nan
  c_m1[np.abs(lon) > 170] = np.nan
  m2[np.abs(lon) > 170] = np.nan
  m1[np.abs(lon) > 170] = np.nan

  # S five point mean
  mu_mag = np.copy(abs_mu)

  # getting the angle and computing betamean
  mu_ang = np.empty(abs_mu.shape)*np.nan
  mu_ang = np.arctan2(muy, mux)
  mu_ang[(mux == 0) & (muy > 0)] = np.pi/2.
  mu_ang[(mux == 0) & (muy < 0)] = 3*np.pi/2.
  mu_ang[(muy == 0)] = 0.

  # shift to get the 4 corners 
  up_shift_ang, down_shift_ang, left_shift_ang, right_shift_ang = fd.four_corner_shift(mu_ang, shift_len=1)
  up_shift_mag, down_shift_mag, left_shift_mag, right_shift_mag = fd.four_corner_shift(mu_mag, shift_len=1)

  # stacking the 5 nearest neighbors for the calculation
  ang_stack = np.dstack((mu_ang, up_shift_ang, down_shift_ang, right_shift_ang, left_shift_ang))
  mag_stack = np.dstack((mu_mag, up_shift_mag, down_shift_mag, right_shift_mag, left_shift_mag))

  # computing the P, Q and n from appendix 2.1
  n = np.nansum(np.double(~np.isnan(ang_stack) & ~np.isnan(mag_stack)))
  sump = np.nansum(mag_stack * np.cos(2*ang_stack), 2)
  sumq = np.nansum(mag_stack * np.sin(2*ang_stack), 2)

  betamean = np.arctan2(sumq, sump) * .5

  c_betamean[np.abs(lat) > 70] = np.nan
  c_betamean[np.abs(lon) > 170] = np.nan
  betamean[np.abs(lat) > 70] = np.nan
  betamean[np.abs(lon) > 170] = np.nan

  ## Resolve the four outer vectors into the positive s_hat [D_mean, B_mean]
  # shifting the mux and muy to get the 4 corners
  # this overlaps the neighbors to allow us to vector caculate
  up_mux, down_mux, left_mux, right_mux = fd.four_corner_shift(mux, shift_len=1)
  up_muy, down_muy, left_muy, right_muy = fd.four_corner_shift(muy, shift_len=1)

  up_lat, down_lat, left_lat, right_lat = fd.four_corner_shift(lat, shift_len=1)
  up_lon, down_lon, left_lon, right_lon = fd.four_corner_shift(lon, shift_len=1)
  left_lon[:, -1] = left_lon[:, -1] + 360
  right_lon[:, 0] = right_lon[:, 0] - 360

  dy = fd.dist_between_grids(up_lat, up_lon, down_lat, down_lon)
  dx = fd.dist_between_grids(left_lat, left_lon, right_lat, right_lon)

  # computing the primes should be done as follows
  # down - up --> dy
  # left - right  --> dx
  down_prime = down_mux*np.cos(betamean) + down_muy*np.sin(betamean)
  up_prime = up_mux*np.cos(betamean) + up_muy*np.sin(betamean)

  left_prime = left_mux*np.cos(betamean) + left_muy*np.sin(betamean)
  right_prime = right_mux*np.cos(betamean) + right_muy*np.sin(betamean)

  dsdiv = (down_prime - up_prime)*np.sin(betamean)/dy + (left_prime - right_prime)*np.cos(betamean)/dx

  c_dsdiv[np.abs(lat) > 70] = np.nan
  c_dsdiv[np.abs(lon) > 170] = np.nan
  dsdiv[np.abs(lat) > 70] = np.nan
  dsdiv[np.abs(lon) > 170] = np.nan

  # geostophic thermal advection
  grad_x, grad_y = fd.geo_gradient(lat, lon, h1km)
  rot_param = 4.*np.pi/24./3600.  * np.sin(lat *np.pi/180.)
  rot_param[np.abs(lat) < 10] = np.nan

  vgx = -(9.81/rot_param)*grad_y
  vgy = (9.81/rot_param)*grad_x

  vgx = fd.smooth_grid(vgx, iter=10, center_weight=4)
  vgy = fd.smooth_grid(vgy, iter=10, center_weight=4)
  
  gta = -1*(vgx*gx + vgy*gy)

  c_gta[np.abs(lat) > 70] = np.nan
  c_gta[np.abs(lat) < 30] = np.nan
  gta[np.abs(lat) > 70] = np.nan
  gta[np.abs(lat) < 30] = np.nan
  
  # catherine fronts uncleaned


  lat_edges = np.asarray(lat[:,0])
  lon_edges = np.asarray(lon[0,:])

  lat_div = lat_edges[1] - lat_edges[0]
  lon_div = lon_edges[1] - lon_edges[0]

  lat_edges = lat_edges - lat_div/2.
  lat_edges = np.append(lat_edges, lat_edges[-1]+lat_div)

  lon_edges = lon_edges - lon_div/2.
  lon_edges = np.append(lon_edges, lon_edges[-1]+lon_div)
  
  # warm fronts
  temp_lat = c_flat[c_ftype > 0]
  temp_lon = c_flon[c_ftype > 0]

  wf, _, _ = np.histogram2d(temp_lat, temp_lon, bins=(lat_edges, lon_edges))
  wf = np.double(wf > 0)
  
  # cold fronts
  temp_lat = c_flat[c_ftype < 0]
  temp_lon = c_flon[c_ftype < 0]
  
  cf, _, _ = np.histogram2d(temp_lat, temp_lon, bins=(lat_edges, lon_edges))
  cf = np.double(cf > 0)

  cath_wf = wf
  cath_cf = cf
  
  zc = fd.mask_zero_contour(lat, lon, dsdiv)
  zc[zc == 0] = np.nan
  zc[~(m1_mask & m2_mask)] = np.nan
  my_cf = np.copy(zc)
  my_wf = np.copy(zc)

  my_wf[gta <= 0] = np.nan
  my_cf[gta >= 0] = np.nan
  
  # ############# JJ TEST END   ################

  # computing the simmonds fronts
  f_sim = fd.simmonds_et_al_2012(lat, lon, prev_u850, prev_v850, u850, v850) 

  # computing the hewson fronts using 1km temperature values, and U & V wind speeds at 850
  f_hew, var = fd.hewson_1998(lat, lon, theta1km, H850)
  # f_hew, var = fd.hewson_1998(lat, lon, theta1km, h1km)

  # f_hew, var = fd.hewson_1998(lat, lon, theta850, H850)
  # zc_6, zc_7 = fd.hewson_1998(lat, lon, theta850, H850)
  
  wf_hew = f_hew['wf']
  cf_hew = f_hew['cf']
  cf_sim = f_sim['cf']

  wf_hew[np.isnan(wf_hew)] = 0
  cf_hew[np.isnan(cf_hew)] = 0
  cf_sim[np.isnan(cf_sim)] = 0

  wf = np.copy(wf_hew)
  cf = np.double((cf_hew + cf_sim) > 0)
  # cf = np.copy(cf_hew)

  wf = my_wf
  cf = my_cf
 
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
  # fronts = cf*-10
  fronts[~((fronts == 10) | (fronts == -10))] = np.nan
  plt.subplot(2,1,1)
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

  plt.subplot(2,1,2)
  fronts = cath_wf*10 + cath_cf*-10
  fronts[~((fronts == 10) | (fronts == -10))] = np.nan
  m = Basemap(projection='cyl', urcrnrlat=ulat, llcrnrlat=llat, urcrnrlon=ulon, llcrnrlon=llon)
  # csf = plt.contourf(cath_lon, cath_lat, cath_slp)
  csf = plt.contourf(lon, lat, var)
  # csf = plt.contourf(lon, lat, slp)
  cs = plt.contour(lon, lat, slp, lw=0.5, ls='--', colors='k', levels=np.arange(980, 1100, 5))
  plt.clabel(cs, inline=1., fontsize=10., fmt='%.0f')
  pc = m.pcolormesh(lon, lat, fronts, cmap='bwr')
  m.colorbar(csf)
  m.drawcoastlines(linewidth=0.2)
  plt.axhline(y=0., linewidth=1.0, linestyle='--')
  plt.title('Catherine Fronts')

  plt.savefig('./images/test.png', dpi=300)
  # plt.draw()
  # plt.show(block=False)
  plt.show()
  
  break

  # plt.savefig('./images/test.png', dpi=300.)

# slv2_id.close()
ncid.close()

