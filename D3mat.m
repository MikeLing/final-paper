%%
clc
clear
R=importdata('H:\multislice network\MSNs data\INFOCOM06.txt');

%R(:,3) = [];
%R(:,3) = [];
%R(:,3) = [];
%R(:,3) = [];
%net=FormNet(R);%��������
t_min=min(R(:,3));%��С��ʼʱ��
t_max=max(R(:,4));%������ʱ�� 
t_max;
t_min;
m = t_max - t_min;%�������ݼ��ڵ�ͨѶ��ʱ��
% ��������mit300  info 120s
s=120;%m/1800;%ʱ��Ƭslice��MIT������30�죬ÿ30����һƬ1800 for mit97��%infocom����3�죬3����һƬ(180 for infocom05);s=1ʱ�������AUC

%sigcomm2009
%T = round((t_max - t_min)/s)
QS=[]; 
A = 0;%��Ƭ�ڽӾ���
TLP=0;
AA=0; %test of A����
A_wei = 0;%��Ƭ�ڽӾ���ʱ�����Ӽ�Ȩ��
a=0.995;% ʱ��Ƭ����
lambda=0.001;%����·��ϵ������������ô��10e-3��
timeSlice=[];%ʱ��Ƭ�ڣ����ӵĸ���
linknum=[];

sampleMat={};%ϸ�����飬�洢ÿ��ʱ��Ƭ��Ȩֵ
sum_adj_mat={};

%% �˲���Ϊ��ʽѭ������ȡÿ��Slice��STLP
% for i=1:round(s)
%     clear xishu_mat;   
%     TS=t_min;%ʱ��Ƭ��ʼʱ��
%     TE=t_min+(m/s);%ʱ��Ƭ����ʱ��
%     [xishu_mat]=xishu_mat(TS,TE,R);  %����һƬϡ�����
%    %% ��һ������ (����dbn֮ǰ����)
%    
%    
% %    %% ����2���ٽ־����Ȩֵ����
%     timeSlice=length(xishu_mat); %��ȡ���һ��ʱ�����ӵĸ���
%     Adj_Mat = xishu2Adj(xishu_mat,R); %�ٽ־���
%     Wei_Adj_Mat=xishu2Wei(xishu_mat,R);%TODO.�ٽ�Ȩֵ����
%     sum_adj_mat{i}=Adj_Mat;%����ʱ�̵��ٽ־��� ����ȷ��label
% 
%    %% LP(����123��·��)
%     sim=Wei_Adj_Mat*Wei_Adj_Mat;
%     STLP=Wei_Adj_Mat+0.01*sim+0.01*0.01*(sim*sim*sim);%sim ��lp ָ��Ľ�� bad% +wei_adj_mat����������ǰ�ڵ�Եļ����������ӵĿ�����
%     %file2txt(xishu_mat,i);
%     sampleMat{i}=STLP;
%     AA=AA+Adj_Mat;%% AA��ʾ�����ӵĸ�����find L
%     A= A + a^(s-i)*Wei_Adj_Mat;%�ٽ�Ȩֵ����  %!!!������STLP��ȷ���ȽϺ� �����ٽ־���ᵼ��ǰ�󲻷�,���ٽ־���Ԥ��ı� ��һ����CN
%     TLP=TLP+STLP;
%     %A_wei= A_wei + a^(s-i)*Wei_Adj_Mat%ͨ��ʱ�����Ӽ�Ȩ����ٽ�Ȩֵ����
%     %save(['file_',num2str(i),'.mat'],'a','A'); %save file
%     
%     t_min=TE; 

%% ���Ը���ָ���AUC(�ⲿ������ʱʹ�ã�Ϊ�˼��ʱ��Ƭ��С��Ԥ��ָ���ӡ��)(S=1ʱ���쾲̬��AUC)
 TS=t_min;%ʱ��Ƭ��ʼʱ��
 TE=t_min+(m/s);%ʱ��Ƭ����ʱ��
 [xishu_mat]=xishu_mat(TS,TE,R);  %����һƬϡ�����
 Adj_Mat = xishu2Adj(xishu_mat,R); %�ٽ־���
 Wei_Adj_Mat=xishu2Wei(xishu_mat,R);%TODO.�ٽ�Ȩֵ����

if ~all(all(xishu_mat(:,1:2)))
    xishu_mat(:,1:2)=xishu_mat(:,1:2)+1;
end

xishu_mat(:,4)=[];
netAdjW=spconvert1(xishu_mat);
xishu_mat(:,4)=[];
netAdj=spconvert(xishu_mat);

nodenum=length(netAdj);
netAdj(nodenum,nodenum)=0;
netAdj=netAdj-diag(diag(netAdj));
netAdj=spones(netAdj+netAdj');
xlswrite('outputSample\netAdj.xlsx', netAdj);
xlswrite('outputSample\netAdjW.xlsx', netAdjW);
[train,test]=DivideNet(netAdj,0.5);%mayuse adj_mat or testAuc/0.8
[trainw,testw]=DivideNet(netAdjW,0.5);%mayuse adj_mat or testAuc/0.8

% we only need tests as 1 to see if connected or not
nodenum=length(testw);
testw(nodenum,nodenum)=0;
testw=testw-diag(diag(testw));
testw=spones(testw+testw');

[raAUC,raPre]=ra(train,test,netAdj);

[cnAUC,cnPre]=CN(train,test);

[aaAUC,aaPre]=AA_AUC(train,test);

kazeAUC=Katz(train,test,lambda);

kazeWAUC=Katzw(train, test, trainw,testw,lambda);
cnAUC

aaAUC

raAUC

cnPre

aaPre

raPre

kazeAUC

kazeWAUC
% %% ��ȡ��������Ҫ��dbn�����ݼ�
% [sorted]=findL(A,20); %A�Ǿ���timeslice20��ʾ���ʱ�̴��ڼ�������
% %sorted ���տ��ܲ������ӵĽڵ�ԴӸߵ�������  һ����timeSlice20�� (����20��)
% tmp=[];%���ڴ洢����
% for i=1:20
% tmp(i)=A(sorted(i,1),sorted(i,2));
% end
% for i=1:20
% tmp(i)=(tmp(i)-mean(tmp))/std(tmp)
% end
% sliceSize=200;%����ά��

%% ���sample
%sample=outPutSamples(sorted,sampleMat,sliceSize,sum_adj_mat);%������label ��������ʱ�����е�ǰ timeslice����������  timeslice*ʱ������ �ľ���  ---�������ļ�

plot(timeSlice,'oR');
%plotCDF(sample1,sample2,sample3);

