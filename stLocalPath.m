function[thisauc]=stLocalPath(A,train,test,lambda)
    sim=A*A;
    sim=A+sim+lambda*(A*A*A);
    thisauc=CalcAUC(train,test,sim,110);
end
