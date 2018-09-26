"""
The class is used for evaluating the application of link prediction
"""

import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
import utils
import config
import math
class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed,nFeatures,emdFeaturesFilename,FeatWtFilename,modes):
        self.embed_filename = embed_filename  # each line: node_id, embeddings[n_embed]
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_emd(embed_filename, n_node=n_node, n_embed=n_embed)
        self.emdFeatures = utils.read_emd(emdFeaturesFilename, n_node=n_node, n_embed=nFeatures)
        self.FeatureWtFilename=FeatWtFilename
        self.mode=modes


    def eval_link_prediction(self):
        """choose the topK after removing the positive training links
        Args:
            test_dataset:
        Returns:
            accuracy:
        """
        featureWt=utils.readFeatureWt(self.FeatureWtFilename)
        test_edges = utils.read_edges_from_file(self.test_filename)
        test_edges_neg = utils.read_edges_from_file(self.test_neg_filename)
        test_edges.extend(test_edges_neg)



        # may exists isolated point
        score_res = []
        wProduct=[]
        if(self.FeatureWtFilename=="/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/src/GraphGAN/genFeatWeights.txt"):
            #print("yes")
           featureWt=np.multiply(featureWt, featureWt);
        #featureWt=np.multiply(featureWt, featureWt)
        for i in range(len(test_edges)):
            embProduct=np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]])
            wProduct=np.dot(np.multiply(self.emdFeatures[test_edges[i][0]],self.emdFeatures[test_edges[i][1]]), featureWt)
            score_res.append(embProduct+wProduct)
        with open(config.dotProduct, mode="a+") as f:
            f.writelines("emb"+str(embProduct) + "----wproduct----"+str(wProduct) )
            f.writelines("\n")
        test_label = np.array(score_res)
        bar = np.median(test_label)  #
        #bar=0.5
        #test------------------------------------median 
        ind_pos = test_label >= bar
        ind_neg = test_label < bar
        test_label[ind_pos] = 1
        test_label[ind_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0:len(true_label) // 2] = 1
        accuracy = accuracy_score(true_label, test_label)

        #test------------------------------------softmax  
        for i in range(0,len(score_res)):
            score_res[i]=1 / (1 + math.exp(-score_res[i]))
        test_label1 = np.array(score_res)      
        barSoftMax=0.5
        ind_pos1 = test_label1 >= barSoftMax
        ind_neg1 = test_label1 < barSoftMax
        test_label1[ind_pos1] = 1
        test_label1[ind_neg1] = 0
        true_label1 = np.zeros(test_label1.shape)
        true_label1[0:len(true_label1) // 2] = 1
        accuracy1 = accuracy_score(true_label1, test_label1)
        res=[]
        res.append(bar)
        res.append(accuracy)
        res.append(accuracy1)
        return(res)
