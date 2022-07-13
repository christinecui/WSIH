function [recall, precision, rate] = recall_precision(Wtrue, Dhat,  max_hamm)
% recall-precision的改变是根据计算出的haming距离阈值决定的，也就是说haming距离为多小时，才会认为是检索正确
% Input:
%    Wtrue = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhat  = estimated distances
%
% Output:
%
%                  exp. # of good pairs inside hamming ball of radius <= (n-1)
%  precision(n) = --------------------------------------------------------------
%                  exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 

if(nargin < 3)
    % 计算出汉明距离矩阵中最大的汉明距离
    max_hamm = max(Dhat(:));
    min_hamm = min(Dhat(:));
    len = max_hamm - min_hamm;
end
hamm_thresh = min(3,max_hamm);

[Ntest, Ntrain] = size(Wtrue);
% True的统计
total_good_pairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(len,1);
recall = zeros(len,1);
rate = zeros(len,1);

% length(precision)就是max_hamm的大小，表示的是所有节点对之间的最大汉明距离
for n = min_hamm:max_hamm-1
    idx = n - min_hamm + 1;
    % 找到距离小于n的所有位置，j是一个布尔值矩阵，不计算最大汉明距离时的情况
    j = (Dhat<=(n+0.00001));
    %exp. # of good pairs that have exactly the same code
    % 一开始n小，对哈明距离约束很大，统计出汉明距离为真且原始数据为真的点
    % 这里存在一个问题，PR曲线的变化是根据汉明距离的阈值改变的产生的，如果阈值很小，可能会导致出现检索结果为0的情况
    retrieved_good_pairs = sum(Wtrue(j));
    
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = sum(j(:));
    
    % 分子表示的是原始数据为真，且计算后也为真的个数，然后比上总的满足汉明距离阈值的个数，就是精准Precision
    % 需要注意原始数据个数真是人工定义的，所以数量不可能为0，但是当汉明距离阈值很低的时候，可能没有满足检索结果，所以需要加上极小量
    precision(idx) = retrieved_good_pairs/(retrieved_pairs+eps);
    % 分子表示的是原始数据为真，且计算后也为真的个数，然后比上总的原始数据真的个数，就是召回Recall
    recall(idx)= retrieved_good_pairs/total_good_pairs;
    rate(idx) = retrieved_pairs / (Ntest*Ntrain);
end

% The standard measures for IR are recall and precision. Assuming that:
%
%    * RET is the set of all items the system has retrieved for a specific inquiry;
%    * REL is the set of relevant items for a specific inquiry;
%    * RETREL is the set of the retrieved relevant items 
%
% then precision and recall measures are obtained as follows:
%
%    precision = RETREL / RET
%    recall = RETREL / REL 

