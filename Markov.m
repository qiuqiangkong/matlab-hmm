% SUMMARY: Calculate the transition matrix for a sequence.
% AUTHOR:  QIUQIANG KONG, Queen Mary University of London
% Created: 15-09-2015
% Modified: - 
% -----------------------------------------------------------
% Usage:   seq = [1 1 1 1 2 2 2 3 3 3 3 3 3 3 2 2 2 ];
%          A = Markov(seq)
% Remarks: Only integral num is available
% ===========================================================
function A = Markov(seq)
max_v = max(seq);
A = zeros(max_v, max_v);
for i1 = 1:length(seq)-1
    A(seq(i1), seq(i1+1)) = A(seq(i1), seq(i1+1)) + 1;
end
A = mk_stochastic(A);   % normalize according to last dim
end