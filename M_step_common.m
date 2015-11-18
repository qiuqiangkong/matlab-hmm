% SUMMARY:  Update p_start, A
%           This is common for multinominal-HMM, Gauss-HMM, GMM-HMM, etc.
% AUTHOR:   QIUQIANG KONG
% Created:  14-11-2015
% Modified: 17-11-2015 Add annotation
% -----------------------------------------------------------
% input:
%   Gamma   p(zn|X^{r})
%   Ksi     p(zn-1,zn|X^{r})
% output:
%   p_start p(z1)
%   A       p(zn|zn-1)
% ===========================================================
function [p_start, A] = M_step_common(Gamma, Ksi)
    obj_num = length(Gamma);
    Q = size(Gamma{1},2);
    
    % calculate p_start
    p_start_numer = zeros(1,Q);
    for r = 1:obj_num
        p_start_numer = p_start_numer + Gamma{r}(1,:);
    end
    p_start = p_start_numer / sum(p_start_numer);
    
    % calculate A
    A_numer = zeros(Q,Q);
    for r = 1:obj_num
        A_numer = A_numer + reshape(sum(Ksi{r},1), Q, Q);
    end
    A = bsxfun(@rdivide, A_numer, sum(A_numer,2));
end