% SUMMARY: This is a Discrete Hidden Markov code
%          This code is inspired by Murphy's PMTK3 toolbox. Using EM
%          algorithm. 
% AUTHOR:  QIUQIANG KONG, Queen Mary University of London
% Created: 17-09-2015
% Modified: - 
% Ref      Chap 13. <Pattern Analysis and Machine Learning>
% -----------------------------------------------------------
% input
%   Data     cell of data
%   p_start  p(z1), size: Q*1
%   A        p(zn|zn-1), transform matrix, size(Q,Q)
%   mu       p(xn|zn), emission matrix, size(N,Q)
%   iter_num how many time the EM should run
% output
%   p_start  p(z1), size: Q*1
%   A        p(zn|zn-1), transform matrix, size(Q,Q)
%   mu       p(xn|zn), emission matrix, size(N,Q)
% ===========================================================
function [p_start, A, mu] = Dhmm(Data, p_start, A, mu, iter_num)
obj_num = length(Data);
[M,Q] = size(mu);

for k = 1:iter_num    
    sum_p_start = zeros(Q,1);
    sum_ita = zeros(Q,Q);
    sum_mu = zeros(M,Q);
    for j1 = 1:obj_num
        % init data
        X = Data{j1};
        N = size(X,1);
        Xmat = zeros(N,M);
        for i1 = 1:N
            Xmat(i1,X(i1)) = 1;
        end

        % calculate p(xn|zn)
        B = Discrete_p_xn_cond_zn(X, mu);   % p(xn|zn), dim 1: N, dim 2: Q
        
        % E step
        [gamma, ita, loglik] = ForwardBackward(p_start,A,B);

        % M step
        sum_p_start = sum_p_start + gamma(1,:)';
        sum_ita = sum_ita + ita;
        
        sum_mu = sum_mu + Xmat'*gamma;
    end
    p_start = normalise(sum_p_start);
    A = mk_stochastic(sum_ita);
    mu = mk_stochastic(sum_mu')';
end
end

% M step
function [pi,A,mu] = Hmm_m_step(X,gamma,ita,M)
    [N,Q] = size(gamma);
    pi = gamma(1,:) / sum(gamma(1,:));
    numer = reshape(sum(ita(2:end,:,:),1), Q, Q);
    denom = sum(numer, 2);
    A = bsxfun(@rdivide, numer, denom);

    Xmat = zeros(N,M);
    for i1 = 1:N
        Xmat(i1,X(i1)) = 1;
    end
    mu = bsxfun(@rdivide, Xmat'*gamma, sum(gamma,1));
end