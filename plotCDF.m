%%»æÖÆCDFÖ÷º¯Êý
function re = plotCDF(sample1,sample2,sample3)
%cdf
initValue=0;
step=0.1;
endValue1=ceil(max(sample1));
endValue2=ceil(max(sample2));
endValue3=ceil(max(sample3));
endValue=max(endValue1,max(endValue2,endValue3));

[xTime1,yPercentage1]=cdf(initValue,step,endValue,sample1);
[xTime2,yPercentage2]=cdf(initValue,step,endValue,sample2);
[xTime3,yPercentage3]=cdf(initValue,step,endValue,sample3);
 
plot(yPercentage1,xTime1,'xB');
hold on 
plot(yPercentage2,xTime2,'oG');
hold on 
plot(yPercentage3,xTime3,'^R');
xlabel('F(X>=x)')
ylabel('Tensor Value')
