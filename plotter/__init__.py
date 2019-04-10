import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot(lon, lat, data, proj='cyl', label_div=10., ax=None, show=False, title=None, cmap='jet'):

  min_lon = np.nanmin(lon)
  min_lat = np.nanmin(lat)
  
  max_lon = np.nanmax(lon)
  max_lat = np.nanmax(lat)

  if (not ax):
    fig = plt.figure()
    ax = plt.subplot(111)

  m = Basemap(projection=proj, llcrnrlon=min_lon, urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat, ax=ax)
  m.drawcoastlines()
  m.drawparallels(np.arange(min_lat, max_lat, label_div), labels=[True, False, False, False])
  m.drawmeridians(np.arange(min_lon, max_lon, label_div), labels=[False, False, False, True])
  c = m.pcolor(lon, lat, data, cmap=cmap)
  m.colorbar(c, ax=ax)

  if (title):
    plt.title(title)

  if (show):
    plt.show()

  return ax
