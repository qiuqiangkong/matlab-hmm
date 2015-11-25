% THIS FILE IS OBSELETED AND REPLACED BY LogViterbiDecode(...)
% SUMMARY:  using viterbi decode to find the best path
%           V1k = p(x1|z1k)
%           Vnk = max{j} (Vn-1,j * p(znk|zn-1,j) * p(xn|znj))
%           save j every step, this is the path. 
% AUTHOR:   QIUQIANG KONG
% Created:  30-09-2015
% Modified: 17-11-2015 Add annotation
%           21-11-2015 Modify to be easy understood
%           25-11-2015 THIS FILE IS OBSELETED AND REPLACED BY LogViterbiDecode(...)
% -----------------------------------------------------------
% input:
%   p_xn_given_zn  p(z(xn|zn)), size:N*Q
%   p_start        p(z1)
%   A              p(zn|zn-1)
% output:
%   path           seq of states, size:N
% ===========================================================
function path = ViterbiDecode(p_xn_given_zn, p_start, A)
[N,Q] = size(p_xn_given_zn);
path = zeros(N,1);
PATH = zeros(N,Q);
V = zeros(N,Q);

V(1,:) = p_start .* p_xn_given_zn(1,:);
for n = 2:N
    C = bsxfun(@times, bsxfun(@times, A, V(n-1,:)'), p_xn_given_zn(n,:) );
    [V(n,:), PATH(n-1,:)] = max(C, [], 1);
end
[~, path(N)] = max(V(n,:));

for n = N-1:-1:1
    path(n) = PATH(n, path(n+1));
end

end
