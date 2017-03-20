function auc=N_auc_net030p(net_graph,probe,t1)
n=size(net_graph);
net_graph=ones(n)-net_graph;
net_graph=triu(net_graph,1);
[non_exist(:,1),non_exist(:,2)]=find(net_graph);
net_graph(sub2ind(n,probe(:,1),probe(:,2)))=1;
net_graph=ones(n)-net_graph-net_graph'-eye(n);
auc=L_auc_net030p(net_graph,probe,non_exist,alpha1,t1);
end