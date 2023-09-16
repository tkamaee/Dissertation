# Dissertation
#
# This repository contains code used for processing fluorescence data, generating timestamps, and plotting graphs within the MSc project "An investigation into the relevance of a blood brain barrier crossing adeno-associated virus vector for in vivo calcium imaging" 
#  
# The general pipeline for processing data was as follows:
#
# 1) unixx_pixel_intensities (Used to convert avi files to 3D matrices).
# 2) no_TS_dfof (Used to generate df/f matrices, and produce df/f plots without timestamps).
# 3) testy (Used to perform continuous wavelt transform, visual df/f plots overalyed with scalograms, adjust parameters, and generate predicted timestamps).
# 4) 1D_fp2 (Used to produce response plots using generated timestamps and df/f matrices. Takes df/f responses around timestamps, aligns and takes a mean then plots and saves aligned mean responses).
# 5) multiple_conditions_mom_v2 (Used to produce reponse plots from mean aligned responses from sessions under different conditions. Includes fucntion for plotting confidence intervals when a mean of mean response is produced from multiple sessions under same conditions).
#
# Please note that ChatGPT 3.5 (Open AI, https://chat.openai.com) was used to support learning to code in python throughout this project, and was used to debug the above code on several occassion.  
