#!/alt/local/bin python

'''
 Converting Zhe's 4/14/2016 version of training code written in MatLAB to Python.
 This script contains three matlab codes: 'main_Train_ARDs.m', 'Merge_sample.m',
										 & 'main_Select_Train_Trends1_6_part2.m'
 Date: 7/12/2016
 Author: Devendra Dahal, EROS, USGS
 Version 0.0
 Based on: 
	 "Function for merging 8 neighborhood sample grids and train the center grid
	 Prepare data for Training_Strategy.m

	 CCDC 1.4 version - Zhe Zhu, EROS, USGS "	
Usage: Merge_ARD_samples_train_v0-1.py -help 
'''
 
import os, sys, traceback, time,subprocess, cPickle,scipy.io
import numpy as np
from datetime import date
import datetime as datetime
from optparse import OptionParser 
from sklearn.ensemble import RandomForestClassifier 
try:
	from osgeo import gdal
	from osgeo.gdalconst import *
except ImportError:
	import gdal	
	
print sys.version

t1 = datetime.datetime.now()
print t1.strftime("\n%Y-%m-%d %H:%M:%S\n")

## define GDAL raster format driver that will be used later on
##--------Start-----------
imDriver = gdal.GetDriverByName('ENVI')
imDriver.Register()

def ComputeRandomForest(X,Y, trees,FileDir, cngrid,n_times):
	print "......processing randomforest model fit for grid%s" % cngrid
	##number of times the area of a standard Landsat scene
	n_times = n_times*(25/37.0)
	# print n_times
	## only use the first column for training
	Y = Y[:,0]
	## remove disturbed classes 3 and 10 from Y
	'''
	ids_rm = np.nonzero((Y != 3) & (Y != 10)& (Y != 0))
	Y = Y[ids_rm]
	X = X[ids_rm]
	'''
	## number of variables
	x_dim = X.shape[0]
	
	## update class number
	all_class = np.unique(Y)
	
	## update number of class
	n_class = all_class.size
	
	## calculate proportion based # for each class
	prct,bin = np.histogram(Y,n_class)
	prct = prct/float(sum(prct))
	# print prct
	# sys.exit()
	## number of reference for euqal number training
	eq_num = np.ceil(20000*n_times) ## total # 
	n_min = np.ceil(600*n_times) ## minimum # 
	n_max = np.ceil(8000*n_times) ## maximum # 

	## intialized selected X & Y training data
	sel_X_trn = []
	sel_Y_trn = []
	for i_cls in range(0,n_class):
	
		## find ids for each land ocver class
		ids = np.where(Y == all_class[i_cls])
		# print (ids[0])
		## total # of referece pixels to permute
		tmp_N = ids[0].size
		# print tmp_N
		## random permute the reference pixels	
		tmp_rv = np.random.permutation(range(tmp_N))

		## adjust num_prop based on proportion
		adj_num = np.ceil(eq_num*prct[i_cls])
		
		## adjust num_prop based on min and max
		if adj_num < n_min:
			adj_num = n_min
		elif adj_num > n_max:
			adj_num = n_max
		
		if tmp_N > adj_num:
			tot_n = adj_num
		else:
			tot_n = tmp_N
		# print tmp_rv[1:int(tot_n)]
		# print ids
		## permutted ids
		rnd_ids = ids[0][tmp_rv[1:int(tot_n)]]

		# print Y[rnd_ids][0:10]
		
		## X_trn and Y_trn
		Y_rnd = Y[rnd_ids].tolist()
		X_rnd = X[rnd_ids,:].tolist()
		# sys.exit()
		
		sel_X_trn = sel_X_trn + X_rnd
		sel_Y_trn = sel_Y_trn + Y_rnd
					
	## log for CCDC Train paramters and versions
	## report only for the first task

	# print len(sel_X_trn)
	# print len(sel_Y_trn)
	modelRF = RandomForestClassifier(n_estimators = trees)
	modelRF = modelRF.fit(sel_X_trn,sel_Y_trn)
	
	### saving as out file
	RFfile = FileDir + os.sep + 'modelRF_py.dump'
	with open(RFfile, 'wb') as f:
		cPickle.dump(modelRF, f)
		
	### saving as .npy file for python	
	RFfile2 = FileDir + os.sep + 'modelRF_py.npy'
        #print "RFfile2=",RFfile2
	np.save(RFfile2, modelRF)   # X is an array
	
	'''
	### saving as .mat file for matlab	
	RFfile1 = FileDir + os.sep + 'modelRF_py.mat'
	h5f = h5py.File(RFfile1, 'w')
	h5f.create_dataset('modelRF', data=modelRF)
	# scipy.io.savemat(RFfile1,mydata)
	'''
	# print modelRF[0]
	# mod = modelRF.fit_transform(sel_X_trn,sel_Y_trn)
	# print mod.shape
	
	print 'model fit completed and 2 version of files saved:\n %s & %s......\n' %('modelRF_py.dump', 'modelRF_py.npy')

def allCalc(nfolder,tree):
	try:			
                 ### Reading TSFitMap for ARD rows and loading as python arrays
                 Datafile_x = nfolder + os.sep + 'Xs.txt'
                 x_data = np.loadtxt(Datafile_x, delimiter =',')
                 Datafile_y = nfolder + os.sep + 'Ys.txt'
                 y_data = np.loadtxt(Datafile_y, delimiter =',')
                 gr_len = 1 
                 cngrid = 11               

                 ## Call defination to compute random forest
                 ## code of main_Select_Train_Trends##_part2.m 
                 ComputeRandomForest(x_data, y_data, tree,nfolder,cngrid,gr_len)
			
	except:
		print "Processed halted on the way."
		print traceback.format_exc()
def main():
	parser = OptionParser()

   # define options
	parser.add_option("-i", dest="in_Folder", help="(Required) Location of input data and place to save output")
	parser.add_option("-t", dest="ntrees", default = 500, help="number of trees, default is 500")
	# parser.add_option("-c", dest="num_cols", help="(Required) number of cols to get image dimension")
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
		allCalc(ops.in_Folder, ops.ntrees)  

if __name__ == '__main__':

	main()
	
t2 = datetime.datetime.now()
print t2.strftime("%Y-%m-%d %H:%M:%S")
tt = t2 - t1
print "\nProcessing time: " + str(tt) 
