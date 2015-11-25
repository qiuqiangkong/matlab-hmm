% SUMMARY:  using log viterbi decode to find the best path. Modified from
%           ViterbiDecode.m
%           ln V1k = ln p(x1|z1k) + ln p(z1k)
%           Vnk = max{j} (ln Vn-1,j + ln p(znk|zn-1,j) + ln p(xn|znj))
%           save j every step, this is the path. 
% AUTHOR:   QIUQIANG KONG
% Created:  21-11-2015
% -----------------------------------------------------------
% input:
%   logp_xn_given_zn  ln p(z(xn|zn)), size:N*Q
%   p_start           p(z1)
%   A                 p(zn|zn-1)
% output:
%   path              seq of states, size:N
% ===========================================================
function path = LogViterbiDecode(logp_xn_given_zn, p_start, A)
[N,Q] = size(logp_xn_given_zn);
path = zeros(N,1);
PATH = zeros(N,Q);

logV = zeros(N,Q);
logV(1,:) = log(p_start) + logp_xn_given_zn(1,:);
for n = 2:N
    C = bsxfun(@plus, bsxfun(@plus, log(A), logV(n-1,:)'), logp_xn_given_zn(n,:) );
    [logV(n,:), PATH(n-1,:)] = max(C, [], 1);
end
[~, path(N)] = max(logV(n,:));

for n = N-1:-1:1
    path(n) = PATH(n, path(n+1));
end

end