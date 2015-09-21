function [p_start, A, Emis] = Chmm_gauss_mix(Data, p_start, A, Emis, varargin)
iter_num = 10;          % the maximum of EM iteration
cov_type = 'diag';      % 'full' or 'diag'
cov_thresh = 1e-4;      % the thresh of cov
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
for k = 1:iter_num
    Xtotal = []; gamma_total = []; w_total = [];
    for i1 = 1:data_num
        
        X = Data{i1};
        N = size(X,1);
        
        % get Ob
%         Ob = Gauss_p_xn_cond_zn(X, Emis);
        Ob = Gaussmix_p_xn_cond_zn(X, Emis);    % p(xn|zn), size: N*Q
        
        
        
        % E
        [gamma, ita, loglik] = ForwardBackward(p_start,A,Ob);

        % M
        
        
        sum_p_start = sum_p_start + gamma(1,:)';
        sum_ita = sum_ita + ita;
        
        Xtotal = [Xtotal; X];
        gamma_total = [gamma_total; gamma];
        
        w = Gaussmix_p_wm_cond_xn(X, Emis);     % p(wn=m|xn), size: N*M
        w_total = [w_total; w];
        
    end

    p_start = normalise(sum_p_start)
    A = mk_stochastic(sum_ita);
    
    Emis = UpdateGaussMixPara(Xtotal,gamma_total, w_total,cov_type,cov_thresh);
end
end

% =====
function Ob = Gaussmix_p_xn_cond_zn(X, Emis)
N = size(X,1);
Q = size(Emis.mu,3);
Ob = zeros(N,Q);
for i1 = 1:Q
    Ob(:,i1) = Gmmpdf(X, Emis.pi(:,i1), Emis.mu(:,:,i1), Emis.Sigma(:,:,:,i1));
end
end

function w = Gaussmix_p_wm_cond_xn(X, Emis)
N = size(X,1);
[p,M,Q] = size(Emis.mu);
w = zeros(N,M,Q);
p_mat = zeros(N,M,Q);
for i1 = 1:Q
    for i2 = 1:M
        p_mat(:,i2,i1) = mvnpdf(X, Emis.mu(:,i2,i1)', Emis.Sigma(:,:,i2,i1));
    end
end

for i1 = 1:Q
    numer = bsxfun(@times, p_mat(:,:,i1), Emis.pi(:,i1)');
    denor = sum(numer, 2);
    w(:,:,i1) = bsxfun(@rdivide, numer, denor);
end
end

% M
function Emis = UpdateGaussMixPara(X,gamma,w,cov_type,cov_thresh)
% [N,Q] = size(gamma);
% for i1 = 1:Q
%     Emis{i1}.mu = X' * gamma(:,i1) / sum(gamma(:,i1));
%     diff = bsxfun(@minus, X, Emis{i1}.mu');
%     Emis{i1}.Sigma = bsxfun(@times, gamma(:,i1), diff)' * diff / sum(gamma(:,i1));
%     if (cov_type=='diag')
%         Emis{i1}.Sigma = diag(diag(Emis{i1}.Sigma));
%     end
%     if max(Emis{i1}.Sigma(:)) < cov_thresh    % prevent cov from being too small
%         Emis{i1}.Sigma = cov_thresh * eye(size(Emis{i1}.Sigma, 1));
%     end
% end
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
        Emis.pi(i2,i1) = sum(gamma_mul_w) / N;
        Emis.mu(:,i2,i1) = (sum(bsxfun(@times, X, gamma_mul_w), 1) / sum(gamma_mul_w))';
        x_minus_mu = bsxfun(@minus, X, Emis.mu(:,i2,i1)');
        Emis.Sigma(:,:,i2,i1) = bsxfun(@times, X, gamma_mul_w)' * X / sum(gamma_mul_w);
        tmp = tmp + gamma_mul_w;
        
        if (cov_type=='diag')
            Emis.Sigma(:,:,i2,i1) = diag(diag(Emis.Sigma(:,:,i2,i1)));
        end
        if max(max(Emis.Sigma(:,:,i2,i1))) < cov_thresh    % prevent cov from being too small
            Emis.Sigma(:,:,i2,i1) = cov_thresh * eye(p);
        end
    end
end

Emis.pi = mk_stochastic(Emis.pi')'; % todo 

% sum(Emis.pi(:))
% Emis.mu
% Emis.Sigma
% sum(Emis.pi(:))
% tmp'
% size(gamma_mul_w)
% pause
end