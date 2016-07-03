% SUMMARY:  Calculate probability of gauss mix model
% AUTHOR:   QIUQIANG KONG, Queen Mary University of London
% Created:  19-09-2015
% Modified: 15-11-2015 Modify output size
%           20-11-2015 debug the order of [p,M]
% -----------------------------------------------------------
% input
%   X               input data; size: N*p; dim 1: num of data, dim 2: feature dim
%   pi              prior of mix; dim 1: mix num
%   mu              size: p*M; dim 1: feature dim, dim 2: mix num
%   Sigma           size: p*p*M; dim 1,2: feature dim, dim 3: mix num
% output
%   probs           probability of input data, size: N*1
% ===========================================================
function probs = Gmmpdf(X, prior, mu, Sigma)
N = size(X,1);          % num of data
[p,M] = size(mu);       % mix num & feature dim
probs = zeros(N,1);     % init output array
for m = 1:M
    probs = probs + prior(m) * mvnpdf(X, mu(:,m)', Sigma(:,:,m));
end
end