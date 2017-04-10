%%����n��3�е�����ϡ�����S�������ĳ��ʱ�����������������ڵ��Լ��Ƿ�������������1,��������0
function [AdjacencyMatrix]=txt2Matrix(TS,TE,R)
row=size(R,1);
B=[];%B�������������������ʱ�������Ľڵ㼯��3�ж���
p=1;
for i=1:row 
    if R(i,3)>=TS && R(i,4)<=TE     
        B(p,1)=R(i,1);%��һ���д�������Ľڵ�
        B(p,2)=R(i,2);%��һ���д�������Ľڵ�
        B(p,3)=1;%�����д�Žڵ������ĳ���ʱ��
        p=p+1;
    elseif R(i,3)<=TS && R(i,4)>=TS &&  R(i,4)<=TE   
        B(p,1)=R(i,1);%��һ���д�������Ľڵ�
        B(p,2)=R(i,2);%��һ���д�������Ľڵ�
        B(p,3)=1;%�����д�Žڵ������ĳ���ʱ��
        p=p+1;
    elseif R(i,3)>=TS && R(i,3)<=TE && R(i,4)>=TE     
        B(p,1)=R(i,1);%��һ���д�������Ľڵ�
        B(p,2)=R(i,2);%��һ���д�������Ľڵ�
        B(p,3)=1;%�����д�Žڵ������ĳ���ʱ��
        p=p+1;
    elseif R(i,3)<=TS && R(i,4)>=TE
        B(p,1)=R(i,1);%��һ���д�������Ľڵ�
        B(p,2)=R(i,2);%��һ���д�������Ľڵ�
        B(p,3)=1;%�����д�Žڵ������ĳ���ʱ��
        p=p+1;
    end
end
maxNode=max(R(:,2));
row2=size(B,1);

%����һ��maxNode*maxNode���ڽӾ��������ӵĽڵ���������ʾ
AdjacencyMatrix=zeros(maxNode);%��һ���վ��� 
for i=1:row2
    m=B(i,1);
    n=B(i,2);  
    AdjacencyMatrix(m,n)=B(i,3);
    AdjacencyMatrix(n,m)=B(i,3); 
end
% %�������г��Խ���Ԫ���⣬����0Ԫ�ر�Ϊ�����
AdjacencyMatrix((AdjacencyMatrix + eye(maxNode)) == 0) = Inf; 