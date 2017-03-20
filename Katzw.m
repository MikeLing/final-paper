function[thisauc]=Katzw(train, test, trainw,testw,lambda)
    sim=inv(sparse(eye(size(trainw,1)))-lambda*trainw);
    sim=sim-sparse(eye(size(trainw,1)));
    xlswrite('outputSample\simKatzW.xlsx', sim);
    thisauc=CalcAUC(train,test,sim,110);
end