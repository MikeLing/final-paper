%%把稀疏矩阵变成具有时间感知的权值矩阵
function [Adj_Mat] = xishu2Adj(B,R)
maxNode=max(R(:,2));
row2=size(B,1);

%建立一个maxNode*maxNode的邻接矩阵，无连接的节点用无穷大表示
AdjacencyMatrix=zeros(maxNode);%建一个空矩阵 
for i=1:row2
    m=B(i,1);
    n=B(i,2);   
end
% %将矩阵中除对角线元素外，其余0元素变为无穷大
%AdjacencyMatrix((AdjacencyMatrix + eye(maxNode)) == 0) = Inf; 
Adj_Mat = AdjacencyMatrix;