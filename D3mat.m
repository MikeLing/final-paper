%%
clc
clear
R=importdata('H:\multislice network\MSNs data\INFOCOM06.txt');

%R(:,3) = [];
%R(:,3) = [];
%R(:,3) = [];
%R(:,3) = [];
%net=FormNet(R);%书上例子
t_min=min(R(:,3));%最小初始时间
t_max=max(R(:,4));%最大结束时间 
t_max;
t_min;
m = t_max - t_min;%整个数据集节点通讯总时长
% 采样周期mit300  info 120s
s=120;%m/1800;%时间片slice（MIT数据下30天，每30分钟一片1800 for mit97）%infocom数据3天，3分钟一片(180 for infocom05);s=1时考察各个AUC

%sigcomm2009
%T = round((t_max - t_min)/s)
QS=[]; 
A = 0;%单片邻接矩阵
TLP=0;
AA=0; %test of A备用
A_wei = 0;%单片邻接矩阵（时间因子加权）
a=0.995;% 时间片参数
lambda=0.001;%三阶路径系数（论文有这么区10e-3）
timeSlice=[];%时间片内，连接的个数
linknum=[];

sampleMat={};%细胞数组，存储每个时间片的权值
sum_adj_mat={};

%% 此部分为正式循环，获取每个Slice的STLP
% for i=1:round(s)
%     clear xishu_mat;   
%     TS=t_min;%时间片开始时间
%     TE=t_min+(m/s);%时间片结束时间
%     [xishu_mat]=xishu_mat(TS,TE,R);  %生成一片稀疏矩阵
%    %% 归一化处理 (留在dbn之前做吧)
%    
%    
% %    %% 计算2个临街矩阵和权值矩阵
%     timeSlice=length(xishu_mat); %获取最后一个时刻连接的个数
%     Adj_Mat = xishu2Adj(xishu_mat,R); %临街矩阵
%     Wei_Adj_Mat=xishu2Wei(xishu_mat,R);%TODO.临街权值矩阵
%     sum_adj_mat{i}=Adj_Mat;%所有时刻的临街矩阵 用来确定label
% 
%    %% LP(考虑123阶路径)
%     sim=Wei_Adj_Mat*Wei_Adj_Mat;
%     STLP=Wei_Adj_Mat+0.01*sim+0.01*0.01*(sim*sim*sim);%sim 是lp 指标的结果 bad% +wei_adj_mat用来考量当前节点对的继续发生链接的可能性
%     %file2txt(xishu_mat,i);
%     sampleMat{i}=STLP;
%     AA=AA+Adj_Mat;%% AA表示用链接的个数来find L
%     A= A + a^(s-i)*Wei_Adj_Mat;%临街权值矩阵  %!!!这里用STLP来确定比较好 若用临街矩阵会导致前后不符,用临街矩阵预测的边 不一定有CN
%     TLP=TLP+STLP;
%     %A_wei= A_wei + a^(s-i)*Wei_Adj_Mat%通过时间因子加权后的临街权值矩阵
%     %save(['file_',num2str(i),'.mat'],'a','A'); %save file
%     
%     t_min=TE; 

%% 测试各个指标的AUC(这部分是临时使用，为了检测时间片大小对预测指标的印象)(S=1时考察静态的AUC)
 TS=t_min;%时间片开始时间
 TE=t_min+(m/s);%时间片结束时间
 [xishu_mat]=xishu_mat(TS,TE,R);  %生成一片稀疏矩阵
 Adj_Mat = xishu2Adj(xishu_mat,R); %临街矩阵
 Wei_Adj_Mat=xishu2Wei(xishu_mat,R);%TODO.临街权值矩阵

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
% %% 获取排序后的需要做dbn的数据集
% [sorted]=findL(A,20); %A是矩阵，timeslice20表示最后时刻存在几条链接
% %sorted 按照可能产生链接的节点对从高到底排序  一共是timeSlice20个 (先做20对)
% tmp=[];%用于存储分数
% for i=1:20
% tmp(i)=A(sorted(i,1),sorted(i,2));
% end
% for i=1:20
% tmp(i)=(tmp(i)-mean(tmp))/std(tmp)
% end
% sliceSize=200;%样本维度

%% 输出sample
%sample=outPutSamples(sorted,sampleMat,sliceSize,sum_adj_mat);%用来做label 返回所有时间序列的前 timeslice对样本集合  timeslice*时间序列 的矩阵  ---并生成文件

plot(timeSlice,'oR');
%plotCDF(sample1,sample2,sample3);

