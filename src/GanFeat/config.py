import numpy as np
batch_size_dis = 64  # batch size for discriminator
batch_size_gen = 64  # batch size for generator
lambda_dis = 1e-5  # l2 loss regulation factor for discriminator
lambda_gen = 1e-5  # l2 loss regulation factor for generator


lambda_dis_ = 1e-6  # l1 loss regulation factor for discriminator
lambda_gen_ = 1e-6  # l1 loss regulation factor for generator


n_sample_dis = 20  # sample num for generator
n_sample_gen = 20  # sample num for discriminator
update_ratio = 1    # updating ratio when choose the trees
save_steps = 2

lr_dis = 1e-4  # learning rate for discriminator
lr_gen = 1e-4  # learning rate for discriminator

max_epochs = 90  # outer loop number
max_epochs_gen = 30 # loop number for generator
max_epochs_dis = 30  # loop number for discriminator

gen_for_d_iters = 20  # iteration numbers for generate new data for discriminator
max_degree = 0  # the max node degree of the network
model_log = "../log/iteration/"

use_mul = True  # control if use the multiprocessing when constructing trees
load_model = False  # if load the model for continual training
gen_update_iter = 4
window_size = 3
#random_state = np.random.randint(0, 100000)
random_state = 6510
app = "citeseerNew"
train_filename = "../../data/" + app + "/train40" + "/trainciteseer40.txt"
test_filename = "../../data/" + app + "/test60" + "/testciteseer60.txt"
test_neg_filename = "../../data/" + app + "/test60" + "/testNegciteseer60.txt"
#train_neg="/home/raavan/mountedDrives/nvme0n1p1/home/amit/feature_add/2/v32/data/fb/train40/fb_train_neg.txt"
n_embed = 32 # 50
n_features = 3703 #104 changed
n_node = 3312

pretrain_feature_filename = "../../features/" + app + "/features.txt"
##added features wt

pretrain_emd_filename_d = "../../pre_train/" + app + "/citeseer_pre_train_3312_32.emb"
pretrain_emd_filename_g = "../../pre_train/" + app + "/citeseer_pre_train_3312_32.emb"



modes = ["gen", "dis"]

emb_filenames = ["../../pre_train/" + app + "/citeseerNew40_" + modes[0] + "_" +"median" + "_" + str(random_state) + ".emb",
                 "../../pre_train/" + app + "/citeseerNew40_" + modes[1] + "_" +"median" + "_" + str(random_state) + ".emb",
                "../../pre_train/" + app + "/citeseerNew40_" + modes[0] + "_" +"softmax" + "_" + str(random_state) + ".emb",
                "../../pre_train/" + app + "/citeseerNew40_" + modes[1] + "_" +"softmax" + "_" + str(random_state) + ".emb",
                 "../../pre_train/" + app + "/citeseerNew40_" + modes[0] + "_" + str(random_state) + ".emb",
                 "../../pre_train/" + app + "/citeseerNew40_" + modes[1] + "_" + str(random_state) + ".emb",
                 "/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/src/GraphGAN/genFeatWeights1.txt",
                "/home/raavan/mountedDrives/home/ayush/twitterGanFEAT/src/GraphGAN/disFeatWeights1.txt",              "../../pre_train/" + app + "/citeseerNew20_Weights_" + modes[0] + "_" +"median" + "_" + str(random_state) + ".emb",
"../../pre_train/" + app + "/citeseerNew40_Weights_" + modes[1] + "_" +"median" + "_" + str(random_state) + ".emb",
"../../pre_train/" + app + "/citeseerNew40_Weights_" + modes[0] + "_" +"softmax" + "_" + str(random_state) + ".emb",
"../../pre_train/" + app + "/citeseerNew40_Weights_" + modes[1] + "_" +"softmax" + "_" + str(random_state) + ".emb"]





                                    
result_filename = "../../results/" + app + "/citeseerNew40_" +  str(random_state) + ".txt"
dotProduct = "../../results/" + app + "/citeseerNew40_" +  str(random_state) + "dotProduct.txt"
