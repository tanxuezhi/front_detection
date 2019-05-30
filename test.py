#!/usr/bin/env python
import numpy as np
import plotter
import reader
import front_detection as fd
import datetime as dt

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def print_date(date):
  print(' %d-%d-%d %02d:00'%(date.year, date.month, date.day, date.hour))

fd_date = dt.datetime(2007, 1, 1, 6)

in_file = '/mnt/drive1/processed_data/tracks/merra2_tracks/ERAI_%d_cyc.mat'%(fd_date.year)
all_center = reader.read_center_from_mat_file(in_file)
center = all_center.find_centers_for_date(fd_date)
