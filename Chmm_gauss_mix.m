function [p_start, A, Emis, loglik] = Chmm_gauss_mix(Data, p_start, A, Emis, varargin)
iter_num = 10;          % the maximum of EM iteration
cov_type = 'diag';      % 'full' or 'diag'
cov_thresh = 1e-4;      % the thresh of cov
converge = 1 + 1e-4;
for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
    end
end

data_num = length(Data);
Q = length(p_start);
sum_p_start = zeros(Q,1);
sum_ita = zeros(Q,Q);

% EM
loglik = 0;
pre_ll = -inf;
for k = 1:iter_num
    Xtotal = []; gamma_total = []; w_total = [];
    for i1 = 1:data_num
        
        X = Data{i1};
        N = size(X,1);
        
        % get Ob
%         Ob = Gauss_p_xn_cond_zn(X, Emis);
%         Ob = Gaussmix_p_xn_cond_zn(X, Emis);    % p(xn|zn), size: N*Q
        logOb = Gaussmix_logp_xn_cond_zn(X, Emis);
        
        % E
%         [gamma, ita, loglik] = ForwardBackward(p_start,A,Ob);
        [gamma, ita, curr_ll] = ForwardBackward(p_start,A,[],logOb);
        loglik = loglik + curr_ll;
        
        % M
        
        sum_p_start = sum_p_start + gamma(1,:)';
        sum_ita = sum_ita + ita;
        
        Xtotal = [Xtotal; X];
        gamma_total = [gamma_total; gamma];
        
        w = Gaussmix_p_wm_cond_xn(X, Emis);     % p(wn=m|xn), size: N*M
        w_total = [w_total; w];
        
    end

    p_start = normalise(sum_p_start);
    A = mk_stochastic(sum_ita);
    
    Emis = UpdateGaussMixPara(Xtotal,gamma_total, w_total,cov_type,cov_thresh);

    if (loglik-pre_ll<log(converge)) break;
    else pre_ll = loglik; end
    
end
end

% =====
% function Ob = Gaussmix_p_xn_cond_zn(X, Emis)
% N = size(X,1);
% Q = size(Emis.mu,3);
% Ob = zeros(N,Q);
% for i1 = 1:Q
%     Ob(:,i1) = Gmmpdf(X, Emis.pi(:,i1), Emis.mu(:,:,i1), Emis.Sigma(:,:,:,i1));
% end
% end

function logOb = Gaussmix_logp_xn_cond_zn(X, Emis)
[N,p] = size(X);
[p,M,Q] = size(Emis.mu);
logOb = zeros(N,Q);

for i1 = 1:Q
    tmpMat = zeros(N,M);
    for i2 = 1:M
        tmpMat(:,i2) = log(Emis.pi(i2,i1)) + Logmvnpdf(X, Emis.mu(:,i2,i1), Emis.Sigma(:,:,i2,i1));
    end
    logOb(:,i1) = max(tmpMat,[],2);
end
end

function w = Gaussmix_p_wm_cond_xn(X, Emis)
N = size(X,1);
[p,M,Q] = size(Emis.mu);
w = zeros(N,M,Q);
p_mat = zeros(N,M,Q);

for i1 = 1:Q
    tmp_mat = zeros(N,M);
    for i2 = 1:M
        tmp_mat(:,i2) = Logmvnpdf(X, Emis.mu(:,i2,i1), Emis.Sigma(:,:,i2,i1));
    end
    [~,loct] = max(tmp_mat,[],2);
    ind = sub2ind([N,M], 1:N, loct');
    Tmp = zeros(N,M);
    Tmp(ind) = 1;
    w(:,:,i1) = Tmp;
end
end



% M
function Emis = UpdateGaussMixPara(X,gamma,w,cov_type,cov_thresh)

[N,Q] = size(gamma);
[N,M,Q] = size(w);
[N,p] = size(X);
Emis.pi = zeros(M,Q);
Emis.mu = zeros(p,M,Q);
Emis.Sigma = zeros(p,p,M,Q);
tmp = 0;
for i1 = 1:Q
    for i2 = 1:M
        gamma_mul_w = gamma(:,i1) .* w(:,i2,i1);
        Emis.pi(i2,i1) = sum(gamma_mul_w) / sum(w(:,i2,i1));
        Emis.mu(:,i2,i1) = (sum(bsxfun(@times, X, gamma_mul_w), 1) / sum(gamma_mul_w))';
        x_minus_mu = bsxfun(@minus, X, Emis.mu(:,i2,i1)');
        Emis.Sigma(:,:,i2,i1) = bsxfun(@times, x_minus_mu, gamma_mul_w)' * x_minus_mu / sum(gamma_mul_w);
        tmp = tmp + gamma_mul_w;

        
        if (cov_type=='diag')
            Emis.Sigma(:,:,i2,i1) = diag(diag(Emis.Sigma(:,:,i2,i1)));
        end
        if max(max(Emis.Sigma(:,:,i2,i1))) < cov_thresh    % prevent cov from being too small
            Emis.Sigma(:,:,i2,i1) = cov_thresh * eye(p);
        end
    end
end

if M==1
    Emis.pi = ones(1,Q);
else
    Emis.pi = mk_stochastic(Emis.pi')';
end
end