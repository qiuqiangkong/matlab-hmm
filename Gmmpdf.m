% SUMMARY: Calculate probability of gauss mix model
% AUTHOR:  QIUQIANG KONG, Queen Mary University of London
% Created: 19-09-2015
% Modified: - 
% -----------------------------------------------------------
% input
%   X               input data; size: N*p; dim 1: num of data, dim 2: feature dim
%   pi              prior of mix; size: M*1; dim 1: mix num
%   mu              size: p*M; dim 1: feature dim, dim 2: mix num
%   Sigma           size: p*p*M; dim 1,2: feature dim, dim 3: mix num
% output
%   probs           probability of input data, size: N*1
% ===========================================================
function probs = Gmmpdf(X, pi, mu, Sigma)
N = size(X,1);          % num of data
M = length(pi);         % num of mix
probs = zeros(N,1);     % init output array
for i1 = 1:M
    probs = probs + pi(i1) * mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
end
end