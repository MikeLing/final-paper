%%
clc
clear
R=importdata('G:\Users\xf\Desktop\multislice network\MSNs data\INFOCOM06.txt');

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
s=1800;%m/1800;%ʱ��Ƭslice��MIT������30�죬ÿ30����һƬ1800 for mit97��%infocom����3�죬3����һƬ(180 for infocom05);s=1ʱ�������AUC

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

for i=1:round(s)

end