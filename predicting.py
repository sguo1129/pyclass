#!/alt/local/bin python

'''
 Converting Zhe's main_ClassificationPar1_4.m and Class_Line1_4.m codes to Python.
 Date: 6/06/2016
 Author: Devendra Dahal, EROS, USGS
 Version 0.0
 Based on: 
	 "
	 CCDC 1.4 version - Zhe Zhu, EROS, USGS "	
Usage: CCDC_main_classification_v1-2.py -help 
'''
 
import os, sys, traceback, time, subprocess, cPickle,scipy.io
import numpy as np
from datetime import date
import datetime as datetime
from optparse import OptionParser 
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
try:
	from osgeo import gdal
	from osgeo.gdalconst import *
except ImportError:
	import gdal	
	
print sys.version

t1 = datetime.datetime.now()
print t1.strftime("%Y-%m-%d %H:%M:%S\n")

## define GDAL raster format driver that will be used later on
##--------Start-----------
imDriver = gdal.GetDriverByName('ENVI')
imDriver.Register()
##--------End-----------

def GetGeoInfo(SourceDS):
	# NDV 		= SourceDS.GetRasterBand(1).GetNoDataValue()
	cols 		= SourceDS.RasterXSize
	rows 		= SourceDS.RasterYSize
	bands	 	= SourceDS.RasterCount
	# GeoT 		= SourceDS.GetGeoTransform()
	# proj 		= SourceDS.GetProjection()
	# extent		= GetExtent(GeoT, cols, rows)
	
	return cols, rows, bands

def Arr_trans(img,cols,rows, b):
	'''
	Reading raster layer and covering to numpy array
	'''
	img_open = gdal.Open(img)
	band1 = img_open.GetRasterBand(b)
	array = band1.ReadAsArray(0, 0, cols, rows)
	# tras_array = np.transpose(array)
	# array = None
	band1 = None
	return array
	
def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array looks like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	return (arr.reshape(h//nrows, nrows, -1, ncols)
			   .swapaxes(1,2)
			   .reshape(-1, nrows, ncols))	

def Class_Line(Datafile,modelRF,num_c,nbands,anc,ntrees,n_class,ys,ye):
	
	rec_cg = scipy.io.loadmat(Datafile)# scipy use
	rec_cg = rec_cg['rec_cg']
	num_ts = rec_cg.shape[1]
	# print rec_cg[0,1]
	## Matrix of each components
	t_start = rec_cg['t_start']
	t_start = np.array(np.concatenate(t_start.reshape(-1).tolist()))
	# t_start = (np.concatenate(t_start.reshape(-1).tolist()).reshape(t_start.shape)).tolist()[0]

	t_end = rec_cg['t_end']
	t_end = np.array(np.concatenate(t_end.reshape(-1).tolist()))
	
	t_break = rec_cg['t_break']
	t_break = np.array(np.concatenate(t_break.reshape(-1).tolist()))
	# t_end = (np.concatenate(t_end.reshape(-1).tolist()).reshape(t_end.shape)).tolist()[0]
	
	# coefs = rec_cg['coefs']
	rmse = rec_cg['rmse']
	pos = rec_cg['pos']
	
	# print pos.shape
	# pos = np.concatenate(pos.reshape(-1).tolist())
	pos = (np.concatenate(pos.reshape(-1).tolist()).reshape(pos.shape))[-1]

	## initiate Ids
	IDS_rec = np.zeros(shape=(1,num_ts))
	
	## number of maps to be created
	yrs = range(ys,ye, 365)
	n_yrs = len(yrs)
	
	# print IDS_rec.shape
	for i_model in range(0,num_ts):
	
		# print rec_cg[0,i_model]['pos'],
		if not rec_cg[0,i_model]['pos'] == True:
			IDS_rec[0,i_model] = 1
			
	##number of valid curves per line
	num_s = np.size(IDS_rec)
	# print 'num_s', num_s

	##number of bands for ancillary data
	n_anc = np.size(anc, axis=2)	

	##has more than one curves exist for each line
	if num_s > 0:
		## model coefficients
		tmp = rec_cg['coefs']
		# print tmp.shape
		##prepare for classification inputs
		Xclass = np.zeros(shape = (num_s,(num_c+1)*(nbands-1) + n_anc))
		
		## array for storing ancillary data
		array_anc = np.zeros(shape = (n_anc,num_s))
		
		for i_a in range(0,n_anc):
			tmp_anc = anc[:,:,i_a]
			tmp_anc = np.array(np.transpose(tmp_anc).reshape(-1).tolist())
			# print tmp_anc[pos]
			array_anc[i_a,:] = tmp_anc[pos]  
		
		map =  np.zeros(shape = (n_yrs,num_s))
		votes =  np.zeros(shape = (n_yrs,num_s,len(n_class)))
		
		##loading model file to start updating it
		with open(modelRF, 'rb') as f:
			rf = cPickle.load(f)
		rmse = np.transpose(np.array(rmse.tolist()[0]),(2,0,1))[0] 
		# print'what is this?', tmp.shape
		tmp1 = np.array(tmp.tolist()[0])
		
		for i, i_yrs in enumerate(yrs):

			id_vali = np.where(np.logical_and(t_start <= i_yrs, np.logical_or(t_end >= i_yrs, t_break > i_yrs)),1,0)
			id_vali = id_vali.reshape(-1).tolist()
			# id_vali = t_start <= i_yrs & (t_end >= i_yrs | t_break > i_yrs)
			for icol in range(0,num_s):
				if id_vali[icol] == 1:
					## coefficients from the 7 bands				
					i_tmp = tmp1[icol:,][0]	
					i_tmp[0,:] = i_tmp[0,:]+i_yrs*i_tmp[1,:]
					i_tmp = np.array(np.transpose(i_tmp[:]).reshape(-1).tolist())
					# print i_tmp
					arr_anc = np.array(np.transpose(array_anc[:,icol]).tolist())
					rms = np.array(rmse[icol].tolist())	

					## input ready
					# Xclass[icol,:] = np.arange(rmse[((icol-1)*(nbands-1)+1),(nbands-1)*icol],nbands-1,1)).reshape(i_tmp[:],array_anc[:,icol])
					Xclass[icol,:] = np.hstack(([rms],[i_tmp],[arr_anc]))
			map_i = rf.predict(Xclass[id_vali,:])
			votes_i = rf.predict_proba(Xclass[id_vali,:])
			map[i,id_vali] = map_i
			votes[i,id_vali,:] = votes_i
			
		if np.count_nonzero(IDS_rec == 1) > 0:
			IDs_add = np.where(IDS_rec == 1)
			for i in range(0,len(IDs_add)):
				
				rec_cg[IDs_add][i]['class'] = map[:,i]
				
				## largest number of votes
				max_id = np.argmax((votes[:,i,:]), axis=1)				
				max_v1 = np.nanmax((votes[:,i,:]), axis=1)
				
				## make this smallest
				votes[:,i,max_id] = 0
				# np.place(votes[:,i,:], votes[:,i,:] = max_v1, [0])
				# np.put(votes[:,i,:], max_id, 0)

				## second largest number of votes
				max_v2 = np.nanmax(votes[:,i,:], axis=1)

				## provide unsupervised ensemble margin as QA
				rec_cg[IDs_add][i]['classQA'] = 100*(max_v1-max_v2)
	## add a new component "class" to rec_cg
	if np.count_nonzero(IDS_rec == 0) > 0:
		IDs_add = np.where(IDS_rec == 0)

		for i in range(0,len(IDs_add)):
			rec_cg[IDs_add[i]]['class'] = ''
			rec_cg[IDs_add[i]]['classQA'] = ''
	
	# print rec_cg.keys()
	scipy.io.savemat(Datafile,{'rec_cg':rec_cg})
	tmp_anc = None
	map = None
	map_i = None
	votes = None
	votes_i = None
	# sys.exit()

def allCalc(FileDir, num_c, nbands, ntrees, s, e):
	try:
		num_c = int(num_c)
		nbands = int(nbands)
		ntrees = int(ntrees)
		
		# print s, e
		s = s.split('-')
		e = e.split('-')
		ys = date.toordinal(date(int(s[0]),int(s[1]),int(s[2])))+366 ## python date num is lagging one year behind\
		ye = date.toordinal(date(int(e[0]),int(e[1]),int(e[2])))+366 ## compared to matlab datenum function
		# yrs = range(ys,ye,365)
		
		'''
		Start selecting input layers that are required 
		'''
		## selecting training data (this is Land cover trends saved as example_img)
		im_roi 		= FileDir + os.sep + 'example_img'
		
		img_roi = gdal.Open(im_roi)
		ncols, nrows, bands = GetGeoInfo(img_roi) 
		tim_roi = Arr_trans(im_roi,ncols,nrows,1)
		modelRF 	= FileDir + os.sep + 'modelRF_py.dump'
		
		## selecting ancillary dataset that came from NLCD
		AncFolder = FileDir + os.sep + 'ANC'
		if not os.path.exists(AncFolder):
			print 'Folder "%s" with ancillary data could not find.' %AncFolder 
			sys.exit()
		
		im_aspect 	= AncFolder + os.sep + 'aspect'
		im_slope	= AncFolder + os.sep + 'slope'
		im_dem 		= AncFolder +  os.sep + 'dem'
		im_posidex	= AncFolder +  os.sep + 'posidex'
		im_wpi		= AncFolder + os.sep + 'mpw'
		
		## converting all raster to np array				
		asp = Arr_trans(im_aspect,ncols,nrows,1)
		# print anc[0:3]
		slp = Arr_trans(im_slope, ncols,nrows,1)
		dem = Arr_trans(im_dem, ncols,nrows,1)
		pdex = Arr_trans(im_posidex, ncols,nrows,1)
		wpi = Arr_trans(im_wpi, ncols,nrows,1)
		
		im_fmask 	= AncFolder + os.sep + 'Fmask_stat'
		## selectig water, snow, and cloud layer from fmask and converting to array
		wtr = Arr_trans(im_fmask,ncols,nrows,1)
		snw = Arr_trans(im_fmask,ncols,nrows,2)
		cld = Arr_trans(im_fmask,ncols,nrows,3)
		
		tim_roi[tim_roi==3] = 10
		tim_roi[tim_roi==4] = 10
		n_class = np.unique(tim_roi)
		
		anc = np.dstack([asp,slp,dem,pdex,wpi,wtr,snw,cld])
		tim_roi	= None
		asp		= None
		slp 	= None
		dem 	= None
		pdex 	= None
		wpi 	= None
		wtr 	= None
		snw 	= None
		cld 	= None
		# if task == 1:
		# print anc[0:3]
		# print 'anc layer done'
		n_result = FileDir + os.sep + 'TSFitMap'
		if not os.path.exists(n_result):
			print 'Folder "%s" with ancillary data could not find.' %n_result 
			os.mkdir(n_result)
			
		# irows = np.zeros(shape=(1,1))
			
		for j in range(1, nrows):
			Datafile = n_result + os.sep + 'record_change'+str(j)+'.mat'
			if not os.path.exists(Datafile):
				print 'Missing the %sth row!' %j
			else:
				print 'Processing the %sth row!' %j
			Class_Line(Datafile,modelRF,num_c,nbands,anc,ntrees,n_class,ys,ye)
	except:
		print traceback.format_exc()
		
def main():
	parser = OptionParser()

   # define options
	parser.add_option("-i", dest="in_Folder", help="(Required) Location of input data and place to save output")
	parser.add_option("-c", dest="num_coefs",default = 8, help="number of coefficient, default is 8")
	parser.add_option("-b", dest="num_bands", default = 8, help="number of bands, default is 8")
	parser.add_option("-t", dest="ntrees", default = 500, help="number of trees, default is 500")
	parser.add_option("-s", dest="y_start",default = '1985-07-01', help="start year of landsat coverage, default is 1985-1-1")
	parser.add_option("-e", dest="y_end",default = '2014-07-01', help="start year of landsat coverage, default is 2014-7-1")
	
	
	(ops, arg) = parser.parse_args()

	if len(arg) == 1:
		parser.print_help()
	elif not ops.in_Folder:
		parser.print_help()
		sys.exit(1)
	# elif not ops.gt_start:
		# parser.print_help()
		# sys.exit(1)
	# elif not ops.gt_end:
		# parser.print_help()
		# sys.exit(1)

	else:
		allCalc(ops.in_Folder, ops.num_coefs, ops.num_bands, ops.ntrees, ops.y_start, ops.y_end)  
		
		
if __name__ == '__main__':

	main()	

	
t2 = datetime.datetime.now()
print t2.strftime("%Y-%m-%d %H:%M:%S")
tt = t2 - t1
print "\nProcessing time: " + str(tt) 
