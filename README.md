# GAN_With_Content_And_Structure
- This repository is the implementation of the following idea:
  - We  aim  to  learn  better  representations  by  exploiting
  both  content  (or  feature)  information  of  nodes  and  structural
  information  of  the  network.  Our  approach  leverages  generative
  adversarial   networks   to   learn   embedding   for   generator   and
  discriminator in a minimax game. While the generator estimates
  the   neighborhood   of   a   node,   the   discriminator   distinguishes
  between the presence or absence of a link for a pair of node

# Requirements
- The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
 - tensorflow == 1.8.0
 - numpy == 1.14.3
 - tqdm == 4.23.4 (for displaying the progress bar)
 - sklearn == 0.19.1
 
# Files in the folder
- data/: training and test data
- feature/: contains feature for each node. The Features are binary values representing the presence or absence of the features
- pre_train/: pre-trained node embeddings
- results/: evaluation results and the learned embeddings of the generator and the discriminator
- src/: source codes 
- precisionAndRecallEvaluation/: for calculating precision values after best embeddings have been learnt

# Input Format
- Data Files: Each line contains an edge information i.e. a sorce and a destination node. Node id starts from 0 to N-1(where N is the number of nodes in the graph)
- Features files: It contains feature for each of the nodes. Each line contains a node id and a 1/0 entries depending on the presence/absence of a particular feature.


# Basic Usage
- cd src/GanFeat
- python graph_gan.py
