%%创建n行3列的无向稀疏矩阵S，存放在某个时间间隔内相遇的两个节点以及是否相遇，相遇置1,不相遇置0
function [AdjacencyMatrix]=txt2Matrix(TS,TE,R)
row=size(R,1);
B=[];%B矩阵用来存放满足上面时间条件的节点集合3列多行
p=1;
for i=1:row 
    if R(i,3)>=TS && R(i,4)<=TE     
        B(p,1)=R(i,1);%第一二列存放相遇的节点
        B(p,2)=R(i,2);%第一二列存放相遇的节点
        B(p,3)=1;%第三列存放节点相遇的持续时间
        p=p+1;
    elseif R(i,3)<=TS && R(i,4)>=TS &&  R(i,4)<=TE   
        B(p,1)=R(i,1);%第一二列存放相遇的节点
        B(p,2)=R(i,2);%第一二列存放相遇的节点
        B(p,3)=1;%第三列存放节点相遇的持续时间
        p=p+1;
    elseif R(i,3)>=TS && R(i,3)<=TE && R(i,4)>=TE     
        B(p,1)=R(i,1);%第一二列存放相遇的节点
        B(p,2)=R(i,2);%第一二列存放相遇的节点
        B(p,3)=1;%第三列存放节点相遇的持续时间
        p=p+1;
    elseif R(i,3)<=TS && R(i,4)>=TE
        B(p,1)=R(i,1);%第一二列存放相遇的节点
        B(p,2)=R(i,2);%第一二列存放相遇的节点
        B(p,3)=1;%第三列存放节点相遇的持续时间
        p=p+1;
    end
end
maxNode=max(R(:,2));
row2=size(B,1);

%建立一个maxNode*maxNode的邻接矩阵，无连接的节点用无穷大表示
AdjacencyMatrix=zeros(maxNode);%建一个空矩阵 
for i=1:row2
    m=B(i,1);
    n=B(i,2);  
    AdjacencyMatrix(m,n)=B(i,3);
    AdjacencyMatrix(n,m)=B(i,3); 
end
% %将矩阵中除对角线元素外，其余0元素变为无穷大
AdjacencyMatrix((AdjacencyMatrix + eye(maxNode)) == 0) = Inf; 