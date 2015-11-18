function test_hmm
close all
addpath('voicebox')
addpath('/homes/qkong/my_code2015.5-/matlab/gmm')


% ================================
% Gmm test
GmmTest();
pause

% ================================
% DO NOT DELETE!
DhmmTest();
pause

% ================================
% continuous hmm, gauss
ChmmGaussTest();
pause

% ================================
% continuous hmm, gauss mix
ChmmGaussMixTest();
pause
end

% ===========
function GmmTest()
close all
% % use pseudo random for debug
% rng(0) 

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
[pi, mu, Sigma, loglik] = Gmm(X, mix_num, 'cov_type', 'diag', 'cov_thresh', 1e-4, 'restart_num', 1, 'iter_num', 100);

% plot gaussians
for i1 = 1:mix_num
    error_ellipse(Sigma(:,:,i1), mu(:,i1)', 'style', 'r'), hold on
end
end

% ============
function DhmmTest()
close all
% % use pseudo random for debug
% rng(0) 

% generate data
Data{1} = [1 1 1 4 1 1 1 2 2 2 2 2 2 1 2 2 2 2 3 3 3 3 3 1 3 3 3]';
Data{2} = [1 1 2 1 1 1 1 1 2 2 2 3 2 2 2 2 2 3 3 3 3 3 4 3 3 3]';
Data_new = [1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3]';


Q = 3;  % num of states
M = 4;  % num of multinominal

% init p(z1)
tmp = rand(1,Q);
p_start0 = tmp / sum(tmp)

% init p(zn-1, zn), dim 1: Q, dim 2: Q
A0 = [0.8 0.2 0; 
    0 0.8 0.2; 
    0 0 1];

% init para of emission probability
phi0.B = ones(M,Q) / M;

% run discrete hmm
[p_start, A, phi, loglik] = Dhmm(Data, Q, M, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'iter_num', 100);

p_start
A
phi.B
loglik

% Calculate p(X) & vertibi decode
p_xn_given_zn = Discrete_p_xn_given_zn(Data_new, phi);
[~,~, loglik] = ForwardBackward(p_xn_given_zn, p_start, A);
path = ViterbiDecode(p_xn_given_zn, p_start, A)
end

function Data = GenerateData1()
for i1 = 1:2
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end

end

% ============
function ChmmGaussTest()
close all
% % use pseudo random for debug
% rng(0) 

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

[p_start, A, phi, loglik] = ChmmGauss(Data, Q);
% [p_start, A, Mu, Sigma, loglik] = Chmm_gauss(Data, Q, 'p_start0', p_start0, 'A0', A0, 'cov_type', 'diag');

p_start
A
phi.mu
phi.Sigma

% Calculate p(X) & vertibi decode
p_xn_given_zn = Gauss_p_xn_given_zn(Data{1}, phi);
[~,~, loglik] = ForwardBackward(p_xn_given_zn, p_start, A);
path = ViterbiDecode(p_xn_given_zn, p_start, A);

% plot gaussians
error_ellipse(reshape(phi.Sigma(:,:,1),p,p), phi.mu(:,1)', 'style', 'r'); hold on
error_ellipse(reshape(phi.Sigma(:,:,2),p,p), phi.mu(:,2)', 'style', 'g'); hold on
error_ellipse(reshape(phi.Sigma(:,:,3),p,p), phi.mu(:,3)', 'style', 'k'); hold on

end

function ChmmGaussMixTest()
close all
% % use pseudo random for debug
% rng(0) 

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

% ------------
[p_start, A, phi, loglik] = ChmmGmm(Data, Q, M);
% [p_start, A, phi, loglik] = ChmmGmm(Data, Q, M, 'p_start0', 'p_start0', 'A0', A0, 'phi0', phi0, 'cov_type', 'diag')
p_start
A
phi.B
phi.mu
phi.Sigma

color = {'r', 'g', 'k'}
for q = 1:Q
    for m = 1:M
        error_ellipse(phi.Sigma(:,:,m,q), phi.mu(:,m,q), 'style', color{q}); hold on
    end
end

end

