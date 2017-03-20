%%
clc
clear
R=importdata('G:\Users\xf\Desktop\multislice network\MSNs data\INFOCOM06.txt');

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
s=1800;%m/1800;%时间片slice（MIT数据下30天，每30分钟一片1800 for mit97）%infocom数据3天，3分钟一片(180 for infocom05);s=1时考察各个AUC

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

for i=1:round(s)

end