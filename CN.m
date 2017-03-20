function[thisauc,thispre]=CN(train,test)
    sim=train*train;
    thisauc=CalcAUC(train,test,sim,110);
    thispre=calcPrecision(test,sim);
end