function[precision]=calcPrecision(testLabel,predictLabel)
testLabel=full(testLabel);
tL=testLabel(:);
predictLabel=full(predictLabel);
pL=predictLabel(:);
p = 0;
pLzhong=round(size(sort(pL),1)/2);
plsort=sort(pL);
middleValue=plsort(pLzhong);
for i=1:length(tL);
    if tL(i)>0
        if pL(i)>middleValue
            p = p+1;
        end
    end
end
a=sort(pL);
oneCount = find(predictLabel>a(1000));
precision =0;
if ~isempty(oneCount)
    precision = p/size(oneCount,1);
end

end