%%输出每个节点对的mat文件 作为基础样本

function  outPutSamples(sorted,sampleMat)
    for i=1:length(sorted)
       node=[];%对应节点
       node=sorted(i,:);
       for t=1:length(sampleMat)
            otp=[];
            otp(t)=sampleMat{i}(node(1),node(2)); 
       end    
   
    end
    