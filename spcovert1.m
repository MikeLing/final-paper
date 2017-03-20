function S = spcovert1( D )
%SPCOVERT1 此处显示有关此函数的摘要
%   此处显示详细说明
if ~issparse(D)
    [~,na] = size(D);
    if na == 4
       S = sparse(D(:,1),D(:,2),D(:,4));
    else
       error(message('MATLAB:spconvert:WrongArraySize'))
    end
else
    S = D;
end

end

