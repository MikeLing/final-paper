function[thisauc,thispre]=aa_stlp(STLP,train,test)
    train1=STLP./repmat(log(sum(STLP,2)),[1,size(STLP,1)]);
    train1(isnan(train1))=0;
    train1(isinf(train1))=0;
    
    sim=train*train1;clear train1;
    thispre=calcPrecision(test,sim);
    thisauc=CalcAUC(train,test,sim,110);
end