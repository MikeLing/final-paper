%%���ÿ���ڵ�Ե�mat�ļ� ��Ϊ��������

function  outPutSamples(sorted,sampleMat)
    for i=1:length(sorted)
       node=[];%��Ӧ�ڵ�
       node=sorted(i,:);
       for t=1:length(sampleMat)
            otp=[];
            otp(t)=sampleMat{i}(node(1),node(2)); 
       end    
   
    end
    