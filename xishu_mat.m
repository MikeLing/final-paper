function [xishu_mat]=xishu_mat(TS,TE,R)
row=size(R,1);
B=[];%B�������������������ʱ�������Ľڵ㼯��3�ж���
p=1;
for i=1:row 
    if (R(i,3)>=TS && R(i,4)<=TE)||(R(i,3)<=TS && R(i,4)>=TS &&  R(i,4)<=TE)||(R(i,3)>=TS && R(i,3)<=TE && R(i,4)>=TE)||(R(i,3)<=TS && R(i,4)>=TE)
         B(p,1)=R(i,1);%��һ���д�������Ľڵ�
         B(p,2)=R(i,2);%��һ���д�������Ľڵ�
         B(p,3)=1;%�����д��ϡ�����ֵ
         B(p,4)=(R(i,4)-R(i,3)*1.0);%�����д�Žڵ�����ӵĳ���ʱ��
         B(p,5)= exp((R(i,4)-R(i,3))/120); %�������д�Žڵ���Ȩ��
         p=p+1; 
        
    end 
end
if isempty(B)
    xishu_mat=[];
    return;
end
A=[];
p=1; %����A��ָ��
i=1; %����B��ָ��
k=1; %����Ƶ�β���
[x,y]=size(B);
A(p,1)=B(i,1);
A(p,2)=B(i,2);
A(p,3)=B(i,3);
A(p,4)=B(i,4);%persist time
A(p,5)=B(i,5);%weight

i=i+1;
for i=2:x    
    if A(p,1)==B(i,1) && A(p,2)==B(i,2)%�ж��Ƿ�ͬһ�Խڵ��
        k=k+1;
        A(p,4)=max(A(p,4),B(i,4));
        A(p,5)=A(p,5)+B(i,5);
    else 
        k=1;
        p=p+1;
        A=[A;[0,0,0,0,0]];%Ϊ����A���һ�У�¼��δ�ظ���Ԫ��
        A(p,1)=B(i,1);
        A(p,2)=B(i,2);
        A(p,3)=B(i,3);
        A(p,4)=B(i,4);
        A(p,5)=B(i,5);
    end
end
xishu_mat = A;
end