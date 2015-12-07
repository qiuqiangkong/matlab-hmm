% SUMMARY:  This is a Discrete Hidden Markov code
%           This code is inspired by Murphy's PMTK3 toolbox. Using EM
%           algorithm. Details are from PRML
% AUTHOR:   QIUQIANG KONG, Queen Mary University of London
% Created:  17-09-2015
% Modified: 17-11-2015
% Ref      Chap 13. <Pattern Analysis and Machine Learning>
% -----------------------------------------------------------
% input
%   Data      cell of data
%   state_num state num
%   mix_num   multinominal num
% varargin input:
%   p_start0  p(z1), size: Q*1
%   A         p(zn|zn-1), transform matrix, size: Q*Q
%   phi0:     emission probability para 
%       B       p(xn|zn), emission matrix, size: p*Q
%   iter_num  how many time the EM should run (default: 100)
%   converge  (default: 1+1e-4)
% output
%   p_start  p(z1), dim 1: Q
%   A        p(zn|zn-1), transform matrix, size: Q*Q
%   mu       p(xn|zn), emission matrix, size: p*Q
% ===========================================================
function [p_start, A, phi, loglik] = Dhmm(Data, state_num, mix_num, varargin)
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'p_start0'
            p_start = varargin{i1+1};
        case 'A0'
            A = varargin{i1+1};
        case 'phi0'
            phi = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
        case 'converge'
            converge = varargin{i1+1};
    end
end
Q = state_num;
M = mix_num;
if (~exist('p_start'))
    tmp = rand(1,Q);
    p_start = tmp / sum(tmp);
end
if (~exist('A'))
    tmp = rand(Q,Q);
    A = bsxfun(@rdivide, tmp, sum(tmp,2));
end
if (~exist('phi'))
    phi.B = ones(M,Q) / M;
end
if (~exist('iter_num'))
    iter_num = 100;
end
if (~exist('converge'))
    converge = 1 + 1e-4;
end

obj_num = length(Data);     % sequences num
[M,Q] = size(phi.B);        % multimominal num, state num
pre_ll = -inf;

for k = 1:iter_num
    % E STEP
    for r = 1:obj_num
        logp_xn_given_zn = Discrete_logp_xn_given_zn(Data{r}, phi);
        [LogGamma{r}, LogKsi{r}, Loglik{r}] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    end
    
    % convert loggamma to gamma, logksi to ksi, substract the max
    [Gamma, Ksi] = UniformLogGammaKsi(LogGamma, LogKsi);

    % M STEP common
    [p_start, A] = M_step_common(Gamma, Ksi);
    
    % M STEP for Multinominal distribution
    B_numer = zeros(M,Q);
    B_denom = zeros(1,Q);
    for r = 1:obj_num
        xr = Data{r};
        Nr = length(xr);
        Xr = zeros(Nr,M);
        Xr(sub2ind([Nr,M], 1:Nr, xr')) = 1;
        B_numer = B_numer + Xr' * Gamma{r};
        B_denom = B_denom + sum(Gamma{r},1);
    end
    phi.B = bsxfun(@rdivide, B_numer, B_denom);
    
    % calculate loglik
    loglik = 0;
    for r = 1:obj_num
        loglik = loglik + Loglik{r};
    end
    if (loglik-pre_ll<log(converge)) break;
    else pre_ll = loglik; end
end