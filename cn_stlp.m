function[thisauc,thispre]=cn_stlp(STLP,train,test)
    sim=STLP*STLP;
    thispre=calcPrecision(test,sim);
    thisauc=CalcAUC(train,test,sim,110);
end