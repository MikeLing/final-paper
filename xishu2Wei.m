%%��ϡ������ɾ���ʱ���֪��Ȩֵ����
function [Adj_Mat] = xishu2Adj(B,R)
maxNode=max(R(:,2));
row2=size(B,1);

%����һ��maxNode*maxNode���ڽӾ��������ӵĽڵ���������ʾ
AdjacencyMatrix=zeros(maxNode);%��һ���վ��� 
for i=1:row2
    m=B(i,1);
    n=B(i,2);   
end
% %�������г��Խ���Ԫ���⣬����0Ԫ�ر�Ϊ�����
%AdjacencyMatrix((AdjacencyMatrix + eye(maxNode)) == 0) = Inf; 
Adj_Mat = AdjacencyMatrix;