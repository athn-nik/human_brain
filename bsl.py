#!/usr/bin/env python
'''
Multiple Linear regression to obtain weights for prediction
inputs: semantic features vectors for nine participants 
        P1-P9
output: cvi activation of voxel v for intermediate semantic feature i
(1-25) sensor-motor verbs  
'''

#from extr_page import noun,sem_feat
import timeit
import scipy.io
from sklearn import linear_model
from scipy.stats.stats import pearsonr
import numpy as np
import itertools
from scipy import spatial
import glob
import sys
import pickle
import re
import heapq
import matlab.engine
import matplotlib
import random
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
from manual_tools import ridge_mod
from RidgeGradientDescent import Ridge

########################################################################
##############Cosine similarity computation for evaluation##############
########################################################################
import numpy as np

def evaluation(i1,p1,i2,p2,metric):
 #print("Cosine Similarity Calculation...")
  #Normalize vectors
 '''i1[:]= [x*x for x in i1]
 magni1=np.sum(i1)
 i1[:]= [x/magni1 for x in i1]
 i2[:]= [x*x for x in i2]
 magni2=np.sum(i2)
 i2[:]= [x/magni2 for x in i2]
 p1[:]= [x*x for x in p1]
 magnp1=np.sum(p1)
 p1[:]= [x/magnp1 for x in p1]
 p2[:]= [x*x for x in p2]
 magnp2=np.sum(p2)
 p2[:]= [x/magnp2 for x in p2]'''
 #print(spatial.distance.cosine(p1,i2),spatial.distance.cosine(p2,i1),spatial.distance.cosine(i2,p2),spatial.distance.cosine(i1,p1))
 if metric=='cosine':
  bad=2-spatial.distance.cosine(p1,i2)-spatial.distance.cosine(p2,i1)
  good=2-spatial.distance.cosine(i2,p2)-spatial.distance.cosine(i1,p1)
 elif metric=='pearson':
  bad=scipy.stats.pearsonr(p1,i2)+scipy.stats.pearsonr(p2,i1)
  good=scipy.stats.pearsonr(i2,p2)+scipy.stats.pearsonr(i1,p1)
 else:
  print("You have given wrong parameter regarding similarity metric!")
  print("give pearson or cosine")
  sys.exit()

 if (bad<good):
  return 1
 else :
  return 0
 return random.choice([True, False])

if __name__ == '__main__':

 ###################################################################
 ######Load semantic features and handle execution requests#########
 ###################################################################

 #for every participant \
 f=open('semantic_features.pkl', 'rb')
 sem_feat=pickle.load(f)
 #print(sem_feat)
 noun=pickle.load(f)
 #print(noun)
 test_pairs=set(itertools.combinations(list(range(60)),2))
 test_pairs=list(test_pairs)
 outFile=open("../outputs/bsl_model.txt", 'w') #w for truncating
 #outFile.write("Test Words             Accuracy\n")
 acc=0
 help=re.findall("\d",sys.argv[1])
 if help==[]:
  no_parts=list(range(1,10))
 else:
  no_parts=list(help)
 if (sys.argv[2]=='-tr'):
  train_option=1
 elif (sys.argv[2]=='-notr'):
  train_option=0
 help=float(sys.argv[3])
 var=float(len(test_pairs))
 help=help*var
 test_pairs=test_pairs[0:int(help)]
 #print(help)
 if (len(sys.argv)>4 and sys.argv[4]=='-st_vox'):
  calc_st=1
 elif (len(sys.argv)<=4):
  calc_st=0

#########################################################################
#Start computations for every participant(1-9) for every test pair(1770)#
#########################################################################
 alpha=[]
 for parts in no_parts :
  print("Processing data for Participant "+str(parts))
  mat = scipy.io.loadmat('../data/data-science-P'+str(parts)+'.mat')
  outFile.write("Participant "+str(parts)+"\n")
  outFile.write("Test Words             Cosine similarity\n")
  acc=0
  fsel=1
  for test_words in test_pairs:
   rate=100*((test_pairs.index(test_words))/len(test_pairs))
   #print("%.1f" % rate,end='\r')
   ##############################################################
   ###############Data Split and merge formatting################
   ##############################################################
   
   test_1=noun[test_words[0]]
   i=1
   #print(test_1)
   test_2=noun[test_words[1]]
   #print("Combination of test words are "+str(test_1)+" "+str(test_2))

   #it goes to 2nd trial and accesses i'th voxel
   #trials are 60 concrete nouns*6 times=360
   #extract data and noun for that data from .mat file

   length=len(mat['data'][0].item()[0])
   #trial data are 6x60=360-2x6=348(test words excluded)
   fmri_data_for_trial=np.zeros((348,length))
   fmri_data_raw=np.zeros((360,length))
   noun_for_trial=[]
   test_data1=np.zeros((6,length))
   test_data2=np.zeros((6,length))
   k=0
   j=0
   colToCoord=np.zeros((length,3))
   coordToCol=np.zeros((mat['meta']['dimx'][0][0][0][0],mat['meta']['dimy'][0][0][0][0],mat['meta']['dimz'][0][0][0][0]))

   colToCoord=mat['meta']['colToCoord'][0][0]
   coordToCol=mat['meta']['coordToCol'][0][0]
   t1=0
   t2=0
   for x in range (0,360):
    fmri_data_raw[k,:]=mat['data'][x][0][0]
    k+=1
    if mat['info'][0][x][2][0]==test_1:
     test_data1[t1,:]=mat['data'][x][0][0]
     t1+=1
    elif mat['info'][0][x][2][0]==test_2:
     test_data2[t2,:]=mat['data'][x][0][0]
     t2+=1
    else:
     fmri_data_for_trial[j,:]=mat['data'][x][0][0]
     noun_for_trial=noun_for_trial+[mat['info']['word'][0][x][0]]
     j+=1

# experimenting with matlab and image plotting 
#   for i in range(21764):
#    img[colToCoord[i,0],colToCoord[i,1],colToCoord[i,2]]=fmri_data_raw[1,i]
#   plot.imshow(img[1,:,:],cmap='gray')
#   print(img[1,:,:])
#   plot.show()
#   sys.exit()
#   #print(fmri_data_raw.tolist())
#   #eng = matlab.engine.start_matlab()

#   #d=matlab.double(fmri_data_raw.tolist())
#   #print(type(fmri_data_raw.tolist()))
#   #print(type(fmri_data_raw.tolist()[0][0]))
#   #eng.figure(nargout=0)
#   #eng.hold("on",nargout=0)
##   eng.box("on",nargout=0)

##   eng.imshow(d[0])
##   eng.quit()
##   sys.exit()
##   a = eng.[datals{:}]; 
##   x = eng.cell2mat(a); 
##   y = eng.double(reshape(x,32,32)
##   eng.figure(nargout=0)
##   eng.hold("on",nargout=0)
##   eng.box("on",nargout=0)
##   
##   eng.imshow(y)
   k=0
   tempo=np.zeros((58,6),dtype=int)
   test1_trials=np.zeros((1,6))
   test2_trials=np.zeros((1,6))
   for x in noun:
    if ((x!=test_1) and (x!=test_2)):
     tempo[k,:]=[i for i, j in enumerate(noun_for_trial) if j == x]
     k+=1
    #elif x==test_1: 
    # test1_trials=[i for i, j in enumerate(noun_for_trial) if j == x]
    #else:
    # test2_trials=[i for i, j in enumerate(noun_for_trial) if j == x]
   combs=set(itertools.combinations([0,1,2,3,4,5],2))
   combs=list(combs)

   ########################################################################
   #################Voxel Stability Selection Starts#######################
   ########################################################################

   #print(test_pairs.index(test_words))
   if (fsel):
    print(parts)
    if (calc_st):
     vox=np.zeros((length,6,60))
     fd=open('/home/n_athan/Desktop/diploma/code/stable_voxels/st_vox'+str(parts)+'.pkl','wb')
     #print(fmri_data_for_trial[tempo[0,:],0])
     stab_score=np.zeros((length))
     for x in range(0,length):#voxel
      sum_vox=0
      h=0
      for y in range(0,60):#noun
       if noun[y]==test_1:
        vox[x,0,y]=test_data1[0,x]
        vox[x,1,y]=test_data1[1,x]
        vox[x,2,y]=test_data1[2,x]
        vox[x,3,y]=test_data1[3,x]
        vox[x,4,y]=test_data1[4,x]
        vox[x,5,y]=test_data1[5,x]
        h+=1
       elif noun[y]==test_2:
        vox[x,0,y]=test_data2[0,x]
        vox[x,1,y]=test_data2[1,x]
        vox[x,2,y]=test_data2[2,x]
        vox[x,3,y]=test_data2[3,x]
        vox[x,4,y]=test_data2[4,x]
        vox[x,5,y]=test_data2[5,x]
        h+=1
       vox[x,0,y]=fmri_data_for_trial[tempo[y-h,0],x]
       vox[x,1,y]=fmri_data_for_trial[tempo[y-h,1],x]
       vox[x,2,y]=fmri_data_for_trial[tempo[y-h,2],x]
       vox[x,3,y]=fmri_data_for_trial[tempo[y-h,3],x]
       vox[x,4,y]=fmri_data_for_trial[tempo[y-h,4],x]
       vox[x,5,y]=fmri_data_for_trial[tempo[y-h,5],x]
       # compute the correlation 
      for z in combs:
       sum_vox+=float(np.correlate(vox[x,z[0],:],vox[x,z[1],:]))
      stab_score[x]=sum_vox/15#no of possible correlations
     stab_vox=heapq.nlargest(500,range(len(stab_score)),stab_score.take)
     pickle.dump(stab_vox,fd)
     fsel=0
   else:
    fd=open('/home/n_athan/Desktop/diploma/code/stable_voxels/st_vox'+str(parts)+'.pkl', 'rb')
    stab_vox=pickle.load(fd)
   fd.close()
   #################################################################
   ########Data preproccesing and mean normalization################
   #################################################################
   
   test_data1=np.sum(test_data1,axis=0)/6
   test_data2=np.sum(test_data2,axis=0)/6
   #print(test_data1.shape)
   #test_data1=test_data1[0,stab_vox]
   #test_data2=test_data2[0,stab_vox]
   
   fmri_data_proc=np.zeros((58,500))
   fmri_data_final=np.zeros((58,500))
   for x in range(0,58):
   # fmri_data_final[x,:] = sum(fmri_data_for_trial[tempo[x,0:5],:])
    fmri_data_proc[x,:] =(fmri_data_for_trial[tempo[x,0],stab_vox]+fmri_data_for_trial[tempo[x,1],stab_vox]+fmri_data_for_trial[tempo[x,3],stab_vox]+fmri_data_for_trial[tempo[x,2],stab_vox]+fmri_data_for_trial[tempo[x,4],stab_vox]+fmri_data_for_trial[tempo[x,5],stab_vox])/6
   #proc
   mean_data=(np.sum(fmri_data_proc,axis=0)+test_data1[stab_vox]+test_data2[stab_vox])/60
   fmri_data_final=np.zeros((58,500))
   mean_data=np.tile(mean_data,(58,1))
   fmri_data_final=fmri_data_proc-mean_data
   #for x in range(0,58):
   # fmri_data_final[x,:]=fmri_data_proc[x,:]-mean_data
   test_data1=test_data1[stab_vox]-mean_data[0,:]
   test_data2=test_data2[stab_vox]-mean_data[0,:]
   
   #########################################################################
   ##########################Training section###############################
   #########################################################################
   
   mle_est=np.ones((500,26))#zeros 25
   semantic=np.zeros((58,25))
   sem_feat=np.array(sem_feat)
   temp=np.ones((60,26))
   temp[:,:-1]=sem_feat
   k=0
   for x in noun:
    if ((x!=test_1) and (x!=test_2)) :
     semantic[k,:]=sem_feat[noun.index(x),:]
     k+=1
   bias=[]
   #semantic=np.tile(semantic,(58,1)
   y,X,k,flag=0,0,0,0
   ridge_mod(y,X,k,flag)
   #print(semantic.shape) 
   if (train_option):
    #for x in range(500):
     #for y in range(500):#every voxel
    model = linear_model.Ridge(300,fit_intercept=True)#####Ridge(alpha=0.5)
     #Here we have to do this for 58/60 for all possible combinations!!
    model.fit(semantic,fmri_data_final)
    mle_est=model.coef_ #TODO remove [x,:
    bias=model.intercept_
    bias=np.array(bias)
    #print(bias.shape)
    bias=np.reshape(np.array(bias),(500,1))
    mle_est=np.append(mle_est,bias,1)
     #alpha=regr.alpha_
     #print(regr.alpha_)
    fd=open('./mle_estimates/coeffs'+str(parts)+'.pkl','wb')
    pickle.dump(mle_est,fd)
   else:
    fd=open('./mle_estimates/coeffs'+str(parts)+'.pkl', 'rb')
    mle_est=pickle.load(fd)
   
   #######################################################################
   #####################Evaluation section################################
   #######################################################################
   
   i1=test_data1
   i2=test_data2
   #we want to found the noun 
   #the noun contained in info
   #mat['info'][0][i][2] contains the word i want for what the trial is set 
   idx1=noun.index(test_1)
   idx2=noun.index(test_2)
   sf1=temp[idx1,:]
   sf2=temp[idx2,:]
   p1=np.dot(mle_est,sf1)
   p2=np.dot(mle_est,sf2)
   acc=acc+evaluation(i1,p1,i2,p2,'cosine')
   #outFile.write(str(test_1)+" "+str(test_2)+"             "+str(acc)+"\n")
  #TODO uncomment for proper use CAUTION
  accuracy=acc/(len(test_pairs))
  outFile.write("\n"+"Total Accuracy "+str(accuracy)+"alpha = "+str(alpha)+"\n")
 outFile.close()









