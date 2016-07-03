% SUMMARY:  Demo of using HMM toolbox wrote by KQQ
%           1.GMM, 2.Multinominal-HMM, 3.Gaussian-HMM, 4.GMM-HMM
% AUTHOR:   QIUQIANG KONG
% Created:  25-11-2015
% Modified: 11-12-2015 Add vertibi decode to GMM-HMM
% ===========================================================
function test_hmm
close all
addpath('matlab-gmm')

% pseudo ranom
% rng(0)

choose = 1;     % can be 1, 2, 3 or 4

switch choose
    case 1
        GmmTest(); 
        return;
    case 2
        DhmmTest(); 
        return;
    case 3
        ChmmGaussTest();
        return
    case 4
        ChmmGaussMixTest();
        return
end

end

% ================================
function GmmTest()
% generate data
X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/4, 50);
X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/4, 70);
X3 = mvnrnd([4,0], [0.5, 0; 0, 0.3]/4, 100);
X4 = [10,10;];
X = [X1; X2; X3; X4];

% plot data
scatter(X(:,1), X(:,2), '.'); hold on

% run gmm
mix_num = 6;
[pi, mu, Sigma, loglik] = Gmm(X, mix_num, 'cov_type', 'diag', 'cov_thresh', 1e-1, 'restart_num', 1, 'iter_num', 100);

% plot gaussians
for i1 = 1:mix_num
    error_ellipse(Sigma(:,:,i1), mu(:,i1)', 'style', 'r'), hold on
end
end

% ================================
function DhmmTest()
% generate data
Data{1} = [1 1 1 4 1 1 1 2 2 2 2 2 2 1 2 2 2 2 3 3 3 3 3 1 3 3 3]';
Data{2} = [1 1 2 1 1 1 1 1 2 2 2 3 2 2 2 2 2 3 3 3 3 3 4 3 3 3]';
Data_new = [1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3]';

Q = 3;  % num of states
M = 4;  % num of multinominal

% init p(z1)
tmp = rand(1,Q);
p_start0 = tmp / sum(tmp);

% init p(zn-1, zn), dim 1: Q, dim 2: Q
A0 = [0.8 0.2 0; 0 0.8 0.2; 0 0 1];

% init para of emission probability
phi0.B = ones(M,Q) / M;

% run discrete hmm
[p_start, A, phi, loglik] = Dhmm(Data, Q, M, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'iter_num', 100);

% Calculate p(X) & vertibi decode
logp_xn_given_zn = Discrete_logp_xn_given_zn(Data_new, phi);
[~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
path = LogViterbiDecode(logp_xn_given_zn, p_start, A);

p_start0
A0
p_start
A
phi.B
loglik
path'

end


% ================================
function ChmmGaussTest()
% Generate Data
for i1 = 1:2
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end
Xall = cell2mat(Data');
scatter(Xall(:,1), Xall(:,2), '.'); hold on

Q = 3;  % state num
p = 2;  % feature dim

p_start0 = [1 0 0];
A0 = [0.8 0.2 0; 0 0.8 0.2; 0 0 1];

[p_start, A, phi, loglik] = ChmmGauss(Data, Q);
% [p_start, A, phi, loglik] = ChmmGauss(Data, Q, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'cov_type', 'diag', 'cov_thresh', 1e-1);

% Calculate p(X) & vertibi decode
logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{1}, phi);
[~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
path = LogViterbiDecode(logp_xn_given_zn, p_start, A);

p_start0
A0
p_start
A
phi
loglik
path'

% plot gaussians
error_ellipse(reshape(phi.Sigma(:,:,1),p,p), phi.mu(:,1)', 'style', 'r'); hold on
error_ellipse(reshape(phi.Sigma(:,:,2),p,p), phi.mu(:,2)', 'style', 'g'); hold on
error_ellipse(reshape(phi.Sigma(:,:,3),p,p), phi.mu(:,3)', 'style', 'k'); hold on
end

% ================================
function ChmmGaussMixTest()
% Generate Data
for i1 = 1:2
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end
for i1 = 3:4
    X1 = mvnrnd([2,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([2,4], [0.5, 0; 0, 0.3]/5, 30);
    X = [X1; X2; X3];
    Data{i1} = X;
end
Xall = cell2mat(Data');
scatter(Xall(:,1), Xall(:,2), '.'); hold on

Q = 3;      % state num
M = 2;      % mix num
p = 2;      % feature dim

% Train Gmm-Hmm model
[p_start, A, phi, loglik] = ChmmGmm(Data, Q, M);
% [p_start, A, phi, loglik] = ChmmGmm(Data, Q, M, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'cov_type', 'diag', 'cov_thresh', 1e-1)

% Calculate p(X) & vertibi decode
logp_xn_given_zn = Gmm_logp_xn_given_zn(Data{1}, phi);
[~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
path = LogViterbiDecode(logp_xn_given_zn, p_start, A);

p_start
A
phi
loglik
path'


color = {'r', 'g', 'k'};
for q = 1:Q
    for m = 1:M
        error_ellipse(phi.Sigma(:,:,m,q), phi.mu(:,m,q), 'style', color{q}); hold on
    end
end

end

