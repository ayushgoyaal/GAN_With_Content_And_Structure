import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import utils
#import config
import math

#generator median gen_median
#generator softmax gen_softmax
#discriminator median dis_median
#Discriminator softmax dis_softmax

#initialize---------------------------------change here---------------------------------------------------
#options
result_filename="citeResult_withFeat_40_60.txt"
with open(result_filename, mode="a+") as f:
    f.writelines("gen median\n\n")

embed_filename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/pre_train/citeseerNew/citeseerNew40_gen_median_6510.emb"
#---------------------------------------------------------------------------------------------------------
test_pos_filename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/data/citeseerNew/test60/testciteseer60.txt"
test_neg_filename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/data/citeseerNew/test60/testNegciteseer_60_5x.txt"
n_node= 3312
n_embed=32
emdFeaturesFilename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/features/citeseerNew/features.txt"
nFeatures=3703
#for generator---------------change here------------------------------------------------------------------
#FeatureWtFilename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/results/gplus/v_6_/disFeatWeights2.txt"
FeatureWtFilename="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/pre_train/citeseerNew/citeseerNew20_Weights_gen_median_6510.emb"

#---------------------------------------------------------------------------------------------------------
#read
emd = utils.read_emd(embed_filename, n_node=n_node, n_embed=n_embed)
emdFeatures = utils.read_emd(emdFeaturesFilename, n_node=n_node, n_embed=nFeatures)
featureWt=utils.readFeatureWt(FeatureWtFilename)
test_edges_pos = utils.read_edges_from_file(test_pos_filename)
test_edges_neg = utils.read_edges_from_file(test_neg_filename)

#for generator---------------change here------------------------------------------------------------------
featureWt=np.multiply(featureWt, featureWt)
#---------------------------------------------------------------------------------------------------------
score_res_pos = []
score_res_neg = []
#score_res= []
wProduct=[]
for i in range(len(test_edges_pos)):
    embProduct=np.dot(emd[test_edges_pos[i][0]], emd[test_edges_pos[i][1]])
    wProduct=np.dot(np.multiply(emdFeatures[test_edges_pos[i][0]],emdFeatures[test_edges_pos[i][1]]), featureWt)
    score_res_pos.append(embProduct+wProduct)

for i in range(len(test_edges_neg)):
    embProduct=np.dot(emd[test_edges_neg[i][0]], emd[test_edges_neg[i][1]])
    wProduct=np.dot(np.multiply(emdFeatures[test_edges_neg[i][0]],emdFeatures[test_edges_neg[i][1]]), featureWt)
    score_res_neg.append(embProduct+wProduct)


for i in range(0,len(score_res_pos)):
    score_res_pos[i]=1 / (1 + math.exp(-score_res_pos[i]))

for i in range(0,len(score_res_neg)):
    score_res_neg[i]=1 / (1 + math.exp(-score_res_neg[i]))

test_label_pos = np.array(score_res_pos)
test_label_neg = np.array(score_res_neg)


barRange=[0.91,0.92,0.93,0.95,0.97,0.98]
for barSoftMax in barRange:
    result=[]
    result="barSoftMax"+str(barSoftMax)+"\n"
    ind_trueEdges_pos = test_label_pos >= barSoftMax #correct labels for true edges
    ind_NoEdges_pos= test_label_neg < barSoftMax #correct labels for no edges

    #precision__0
    totalN=len(test_label_neg)
    TN=sum(ind_NoEdges_pos)
    precision=TN/totalN
    result=result+"precision__0="+str(precision)+"\n"
    #precision__1
    totalP=len(test_label_pos)
    TP=sum(ind_trueEdges_pos)
    precision=TP/totalP
    result=result+"precision__1="+str(precision)+"\n"

    ###total Accuracy
    total=(TN+TP)/(totalN+totalP)
    result=result+"total="+str(total)+"\n\n"
    with open(result_filename, mode="a+") as f:
        f.writelines(result)

with open(result_filename, mode="a+") as f:
    f.writelines("     ------------**************-----------------    \n\n")
