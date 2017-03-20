function auc=L_auc_net030p(net_graph,probe,non_exist,alpha1,t1)
L=size(probe,1);
nn=size(net_graph,1);
num_nonexist=size(non_exist,1);

G=(net_graph)*(net_graph);
B=zeros(nn,nn);
for i=1:nn
        for j=1:nn
            if net_graph(i,j)==0&&G(i,j)~=0
                B(i,j)=1;
            end
        end
end
for i=1:nn
    B(i,i)=0;
end
D=net_graph;
net_graph=D*net_graph;
C=B*D.';
B=D*B+C;
auc=zeros(1,length(alpha1));
for ii=1:length(alpha1)
    alpha=alpha1(ii);
    D=(1-2*alpha)*net_graph+alpha*(B);
    for i=1:t1
        k=ceil(rand()*L);j=ceil(rand()*num_nonexist);
        kk1=D(probe(k,1),probe(k,2));
        jj1=D(non_exist(j,1),non_exist(j,2));
        if kk1>jj1
            auc(ii)=auc(ii)+1;
        else
            if kk1==jj1
                auc(ii)=auc(ii)+0.5;
            end
        end
    end
    auc(ii)=auc(ii)/t1;
end
end