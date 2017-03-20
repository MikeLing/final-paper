function[thisauc]=Katz(train,test,lambda)
    sim=inv(sparse(eye(size(train,1)))-lambda*train);
    sim=sim-sparse(eye(size(train,1)));
    thisauc=CalcAUC(train,test,sim,110);
end