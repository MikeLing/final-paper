
function  smp=outPutSamples(sorted,sampleMat)
    smp=[];
    for i=1:length(sorted)
       node=[];%对应节点
       node=sorted(i,:);
       otp=[];
       for t=1:length(sampleMat)           
            otp(t)=sampleMat{t}(node(1),node(2)); 
       end
       smp(i,:)=otp;
    %save(['file_',num2str(i),'.mat'],'a','A'); %save file
    end
    return