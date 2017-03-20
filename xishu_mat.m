function [xishu_mat]=xishu_mat(TS,TE,R)
row=size(R,1);
B=[];%B矩阵用来存放满足上面时间条件的节点集合3列多行
p=1;
for i=1:row 
    if (R(i,3)>=TS && R(i,4)<=TE)||(R(i,3)<=TS && R(i,4)>=TS &&  R(i,4)<=TE)||(R(i,3)>=TS && R(i,3)<=TE && R(i,4)>=TE)||(R(i,3)<=TS && R(i,4)>=TE)
         B(p,1)=R(i,1);%第一二列存放相遇的节点
         B(p,2)=R(i,2);%第一二列存放相遇的节点
         B(p,3)=1;%第三列存放稀疏矩阵值
         B(p,4)=(R(i,4)-R(i,3)*1.0);%第四列存放节点间链接的持续时长
         B(p,5)= exp((R(i,4)-R(i,3))/120); %第五列中存放节点间的权重
         p=p+1; 
        
    end 
end
if isempty(B)
    xishu_mat=[];
    return;
end
A=[];
p=1; %矩阵A行指针
i=1; %矩阵B行指针
k=1; %控制频次参数
[x,y]=size(B);
A(p,1)=B(i,1);
A(p,2)=B(i,2);
A(p,3)=B(i,3);
A(p,4)=B(i,4);%persist time
A(p,5)=B(i,5);%weight

i=i+1;
for i=2:x    
    if A(p,1)==B(i,1) && A(p,2)==B(i,2)%判断是否同一对节点对
        k=k+1;
        A(p,4)=max(A(p,4),B(i,4));
        A(p,5)=A(p,5)+B(i,5);
    else 
        k=1;
        p=p+1;
        A=[A;[0,0,0,0,0]];%为矩阵A添加一行，录入未重复的元素
        A(p,1)=B(i,1);
        A(p,2)=B(i,2);
        A(p,3)=B(i,3);
        A(p,4)=B(i,4);
        A(p,5)=B(i,5);
    end
end
xishu_mat = A;
end