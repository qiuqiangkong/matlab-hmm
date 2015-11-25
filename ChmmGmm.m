% SUMMARY:  Train Gauss-HMM model
% AUTHOR:   QIUQIANG KONG
% Created:  17-11-2015
% Modified: 25-11-2015 Add annotation
% -----------------------------------------------------------
% input:
%   Data        cell of data
%   state_num   state num
%   mix_num     multinominal num
% varargin input:
%   p_start0    p(z1), size: Q*1
%   A           p(zn|zn-1), transform matrix, size: Q*Q
%   phi0:       emission probability para 
%       B         size: M*Q
%       mu        size: p*M*Q
%       Sigma     size: p*p*M*Q
%   iter_num    how many time the EM should run (default: 100)
%   converge    (default: 1+1e-4)
% output
%   p_start  p(z1), dim 1: Q
%   A        p(zn|zn-1), transform matrix, size: Q*Q
%   phi0:       emission probability para 
%       B         size: M*Q
%       mu        size: p*M*Q
%       Sigma     size: p*p*M*Q
% ===========================================================
function [p_start, A, phi, loglik] = ChmmGmm(Data, state_num, mix_num, varargin)
% Init Paras
Q = state_num;
M = mix_num;
p = size(Data{1},2);
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'p_start0'
            p_start = varargin{i1+1};
        case 'A0'
            A = varargin{i1+1};
        case 'phi0'
            phi = varargin{i1+1};
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
        case 'converge'
            converge = varargin{i1+1};
    end
end
if (~exist('p_start'))
    tmp = rand(1,Q);
    p_start = tmp / sum(tmp);
end
if (~exist('A'))
    tmp = rand(Q,Q);
    A = bsxfun(@rdivide, tmp, sum(tmp,2));
end
if (~exist('phi'))
    Xall = cell2mat(Data');
    [prior_, mu_, Sigma_] = Gmm(Xall, M*Q, 'diag');
    tmp = reshape(prior_,M,Q);
    phi.B = bsxfun(@rdivide, tmp, sum(tmp, 1));
    phi.mu = reshape(mu_,p,M,Q);
    phi.Sigma = reshape(Sigma_,p,p,M,Q);
end
if (~exist('iter_num'))
    iter_num = 100;          % the maximum of EM iteration
end
if (~exist('cov_type'))
    cov_type = 'diag';      % 'full' or 'diag'
end
if (~exist('cov_thresh'))
    cov_thresh = 1e-4;      % the thresh of cov
end
if (~exist('converge'))
    converge = 1 + 1e-4;
end

pre_ll = -inf;
obj_num = length(Data);
for k = 1:iter_num
    % E STEP
    for r = 1:obj_num
        logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{r}, phi);
        [LogGamma{r}, LogKsi{r}, Loglik{r}] = LogForwardBackward(logp_xn_given_zn, p_start, A);
        logp_xn_given_vn = Get_logp_xn_given_vn(Data{r}, phi);
        LogIta{r} = CalculateLogIta(logp_xn_given_vn, p_start, A, phi);
    end
    
    % convert loggamma to gamma, logksi to ksi, substract the max
    [Gamma, Ksi] = UniformLogGammaKsi(LogGamma, LogKsi);
    
    % convert logita to ita, substract the max
    Ita = UniformLogIta(LogIta);
    
    % M STEP common
    [p_start, A] = M_step_common(Gamma, Ksi);
    
    % M STEP for Gmm
    % update B
    B_nomer = zeros(M,Q);
    B_denom = zeros(1,Q);
    for r = 1:obj_num
        B_nomer = B_nomer + reshape(sum(Ita{r},1), M, Q);
        B_denom = B_denom + reshape(sum(sum(Ita{r},2),1), 1, Q);
    end
    phi.B = bsxfun(@rdivide, B_nomer, B_denom);
    
    % update mu, Sigma
    mu_numer = zeros(p,M,Q);
    mu_denom = zeros(M,Q);
    for q = 1:Q
        for m = 1:M
            mu_numer = zeros(p,1);
            mu_denom = 0;
            for r = 1:obj_num
                mu_numer = mu_numer + Data{r}' * Ita{r}(:,m,q);
                mu_denom = mu_denom + sum(Ita{r}(:,m,q));
            end
            phi.mu(:,m,q) = mu_numer / mu_denom;
            
            Sigma_numer = zeros(p,p);
            for r = 1:obj_num
                x_diff_mu = bsxfun(@minus, Data{r}, phi.mu(:,m,q)');
                Sigma_numer = Sigma_numer + bsxfun(@times, Ita{r}(:,m,q), x_diff_mu)' * x_diff_mu;
            end
            phi.Sigma(:,:,m,q) = Sigma_numer / mu_denom;
            
            if (cov_type=='diag')
                phi.Sigma(:,:,m,q) = diag(diag(phi.Sigma(:,:,m,q)));
            end
            if min(eig(phi.Sigma(:,:,m,q))) < cov_thresh    % prevent cov from being too small
                phi.Sigma(:,:,m,q) = phi.Sigma(:,:,m,q) + cov_thresh * eye(p);
            end
        end
    end
    
    % loglik
    loglik = 0;
    for r = 1:obj_num
        loglik = loglik + Loglik{r};
    end
    if (loglik-pre_ll<log(converge)) break;
    else pre_ll = loglik; end
end

end

% output: ln p(xn|vn), size: N*M*Q
function logp_xn_given_vn = Get_logp_xn_given_vn(X, phi)
    [N,p] = size(X);
    [M,Q] = size(phi.B);
    logp_xn_given_vn = zeros(N,M,Q);
    for q = 1:Q
        for m = 1:M
            x_minus_mu = bsxfun(@minus, X, phi.mu(:,m,q)');
            logp_xn_given_vn(:,m,q) = -0.5*p*log(2*pi) - 0.5*log(det(phi.Sigma(:,:,m,q))) - 0.5 * sum(x_minus_mu * inv(phi.Sigma(:,:,m,q)) .* x_minus_mu, 2);
        end
    end
end

% output: ln p(vn|xn), size: N*M*Q
function logita = CalculateLogIta(logp_xn_given_vn, p_start, A, phi)
    [N,M,Q] = size(logp_xn_given_vn);
    
    % reserve space
    logc = zeros(N,1);
    logalpha = zeros(N,M,Q);
    logbeta = zeros(N,M,Q);
    logita = zeros(N,M,Q);
    
    Tmp = bsxfun( @plus, log(phi.B) + reshape(logp_xn_given_vn(1,:,:),M,Q), log(p_start) );
    logc(1) = log( sum( sum( exp( Tmp - max(Tmp(:)) ) ) ) ) + max(Tmp(:));
    logalpha(1,:,:) = -logc(1) + Tmp;
    logbeta(N,:,:) = 0;
 
    % calculate c, alpha
    for n = 2:N
        T4 = zeros(M,Q,M,Q);    % dim 1,2: vn-1; dim 3,4: vn
        for q = 1:Q
            for m = 1:M
                T4(:,:,m,q) = logp_xn_given_vn(n,m,q) + log(phi.B) + bsxfun( @plus, reshape(logalpha(n-1,:,:),M,Q), log(A(:,q)') );
            end
        end
        tmp = exp( T4 - max(T4(:)) );
        logc(n) = log( sum(tmp(:)) ) + max(T4(:));
        
        for q = 1:Q
            for m = 1:M
                T2 = bsxfun( @plus, reshape(logalpha(n-1,:,:),M,Q), log(A(:,q)') );
                if isinf(max(T2(:)))
                    logalpha(n,m,q) = -inf;
                else
                    logalpha(n,m,q) = -logc(n) + logp_xn_given_vn(n,m,q) + log(phi.B(m,q)) + log( sum( sum( exp( T2 - max(T2(:)) ) ) ) ) + max(T2(:));
                end
            end
        end
    end
    
    for n = N-1:-1:1
        for q = 1:Q
            for m = 1:M
                T2 = bsxfun( @plus, reshape(logbeta(n+1,:,:),M,Q) + reshape(logp_xn_given_vn(n+1,:,:),M,Q) + log(phi.B), log(A(q,:)) );
                logbeta(n,m,q) = -logc(n+1) + log( sum( sum( exp( T2 - max(T2(:) ) ) ) ) ) + max(T2(:));
            end
        end
    end
    
    % calculate ita
    logita = logalpha + logbeta;
end

% convert logita to ita, substract the max
function Ita = UniformLogIta(LogIta)
    obj_num = length(LogIta);
    Q = size(LogIta{1}, 3);
    for q = 1:Q
        max_ita_ary = zeros(1, obj_num);
        for r = 1:obj_num
            max_ita_ary(r) = max(max(LogIta{r}(:,:,q)));
        end
        max_ita = max(max_ita_ary);
        
        for r = 1:obj_num
            LogIta{r}(:,:,q) = LogIta{r}(:,:,q) - max_ita;
        end
    end
    
    for r = 1:obj_num
        Ita{r} = exp(LogIta{r});
    end
end