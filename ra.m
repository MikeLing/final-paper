function[thisauc,thispre]=ra(train,test,netAdj)
    train1=train./repmat(sum(train,2),[1,size(train,1)]);
    train1(isnan(train1))=0;
    train1(isinf(train1))=0;
    
    sim=train*train1;clear train1;
    thispre=calcPrecision(test,sim);
    thisauc=CalcAUC(train,test,sim,110);
end