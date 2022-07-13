function [recall, precision, rate] = recall_precision(Wtrue, Dhat,  max_hamm)
% recall-precision�ĸı��Ǹ��ݼ������haming������ֵ�����ģ�Ҳ����˵haming����Ϊ��Сʱ���Ż���Ϊ�Ǽ�����ȷ
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
    % ���������������������ĺ�������
    max_hamm = max(Dhat(:));
    min_hamm = min(Dhat(:));
    len = max_hamm - min_hamm;
end
hamm_thresh = min(3,max_hamm);

[Ntest, Ntrain] = size(Wtrue);
% True��ͳ��
total_good_pairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(len,1);
recall = zeros(len,1);
rate = zeros(len,1);

% length(precision)����max_hamm�Ĵ�С����ʾ�������нڵ��֮������������
for n = min_hamm:max_hamm-1
    idx = n - min_hamm + 1;
    % �ҵ�����С��n������λ�ã�j��һ������ֵ���󣬲��������������ʱ�����
    j = (Dhat<=(n+0.00001));
    %exp. # of good pairs that have exactly the same code
    % һ��ʼnС���Թ�������Լ���ܴ�ͳ�Ƴ���������Ϊ����ԭʼ����Ϊ��ĵ�
    % �������һ�����⣬PR���ߵı仯�Ǹ��ݺ����������ֵ�ı�Ĳ����ģ������ֵ��С�����ܻᵼ�³��ּ������Ϊ0�����
    retrieved_good_pairs = sum(Wtrue(j));
    
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = sum(j(:));
    
    % ���ӱ�ʾ����ԭʼ����Ϊ�棬�Ҽ����ҲΪ��ĸ�����Ȼ������ܵ����㺺��������ֵ�ĸ��������Ǿ�׼Precision
    % ��Ҫע��ԭʼ���ݸ��������˹�����ģ���������������Ϊ0�����ǵ�����������ֵ�ܵ͵�ʱ�򣬿���û��������������������Ҫ���ϼ�С��
    precision(idx) = retrieved_good_pairs/(retrieved_pairs+eps);
    % ���ӱ�ʾ����ԭʼ����Ϊ�棬�Ҽ����ҲΪ��ĸ�����Ȼ������ܵ�ԭʼ������ĸ����������ٻ�Recall
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

