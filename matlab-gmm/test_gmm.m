% SUMMARY:  This is a simple demo for using Gmm
% AUTHOR:   QIUQIANG KONG
% Created:  2015.09.20
% Modified: 2015.11.17 Add annoatations
%           2015.12.18 Add usage of Gmmpdf
% ===========================================================
function test_gmm
close all
addpath('voicebox');    % need kmeans in voicebox

% use pesudo ranodm
% rng(0)

% generate data
X = GenerateData1();
scatter(X(:,1), X(:,2), '.'); axis([-2 12 -2 12]); hold on

% run gmm
mix_num = 6;
[prior, mu, Sigma, loglik] = Gmm(X, mix_num);
[prior, mu, Sigma, loglik] = Gmm(X, mix_num, 'cov_type', 'diag', 'cov_thresh', 1e-4, 'restart_num', 1, 'iter_num', 100);

% calculate probability using gmm
probs = Gmmpdf(X, prior, mu, Sigma);

% plot gaussians
for m = 1:mix_num
    error_ellipse(Sigma(:,:,m), mu(:,m), 'style', 'r'), hold on
end
end

% generate data
function X = GenerateData1()
X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/4, 50);
X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/4, 70);
X3 = mvnrnd([4,0], [0.5, 0; 0, 0.3]/4, 100);
X4 = [10,10;];
X = [X1; X2; X3; X4];
end
