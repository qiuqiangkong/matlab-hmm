function test
close all
addpath('/homes/qkong/my_code2015.5-/matlab/toolbox/voicebox')


% ================================
% DO NOT DELETE!
% % Markov test
% Markov([1 1 1 1 2 2 2 3 3 3 1 1 2 2 2 2])

% ================================
% % Gmm test

% GmmTest();

% ================================
% % DO NOT DELETE!
% DhmmTest();
% pause

% ================================
% continuous hmm, gauss
ChmmGaussTest();

% ================================
% continuous hmm, gauss mix
% ChmmGaussMixTest();

% ================================

end

% ===========
function GmmTest()
rng(0)
X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/4, 50);
X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/4, 70);
X3 = mvnrnd([4,0], [0.5, 0; 0, 0.3]/4, 100);
X4 = [10,10;];
X = [X1; X2; X3; X4];
scatter(X(:,1), X(:,2), '.'); hold on
mix = 6;
[pi, mu, Sigma] = Gmm(X, mix, 'cov_type','diag', 'cov_thresh', 1e-2);
for i1 = 1:mix
    error_ellipse(Sigma(:,:,i1), mu(:,i1), 'style', 'r'), hold on
end

probs = Gmmpdf(X, pi, mu, Sigma);



end

% ============
function DhmmTest()
% discrete hmm
Data{1} = [1 1 1 4 1 1 1 2 2 2 2 2 2 1 2 2 2 2 3 3 3 3 3 1 3 3 3]';
Data{2} = [1 1 2 1 1 1 1 1 2 2 2 3 2 2 2 2 2 3 3 3 3 3 4 3 3 3]';

% init pi, A, mu
Q = 3; M = 4; 
rng('default')
p_start0 = mk_stochastic(rand(Q,1));  % p(z1), dim 1: Q
% A0 = mk_stochastic(rand(Q,Q));   % p(zn-1, zn), dim 1: Q, dim 2: Q
A0 = [0.5 0.5 0; 0 0.5 0.5; 0 0 1];
b = [0.1; 0.2; 0.3; 0.4];
mu0 = repmat(b, 1, Q);            % dim 1: M, dim 2: Q
% mu0 = repmat(ones(M,1)/M,1,Q);
iter_num = 10;
[p_start, A, mu] = Dhmm(Data, p_start0, A0, mu0, iter_num);
p_start
A
mu

Ob = Discrete_p_xn_cond_zn(Data{2}, mu);
[gamma, sum_ita, loglik] = ForwardBackward(p_start,A,Ob);
path = ViterbiDecode(p_start,A,Ob);
end

function Data = GenerateData1()
for i1 = 1:10
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([4,0], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end

end

% ============
function ChmmGaussTest()
rng(0)
% X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
% X2 = mvnrnd([2,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
% X3 = mvnrnd([4,0], [0.5, 0; 0, 0.3]/5, 40);
% X4 = mvnrnd([1,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
% X5 = mvnrnd([3,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
% X6 = mvnrnd([5,0], [0.5, 0; 0, 0.3]/5, 40);
% 
% X = [X1; X2; X3; X4; X5; X6];
% Data{1} = [X1;X2;X3];
% Data{2} = [X4;X5;X6];
% scatter(X(:,1), X(:,2), '.'); hold on

Data = GenerateData1();
Xall = cell2mat(Data');
size(Xall)
% scatter(Xall(:,1), Xall(:,2), '.'); hold on

Q = 3;
p = 2;  % feature dim

% init p_start, emission, A
% p_start0 = ones(Q,1) / Q;
% for i1 = 1:Q
%     Emis0{i1}.mu = zeros(p,1);
%     Emis0{i1}.Sigma = 0.1*eye(p);
% end

for i1 = 1:Q-1
    A0(i1,i1:i1+1) = 0.5;
end
A0(Q,Q) = 1;


% p_start0 = normalise(rand(Q,1));
Xall = cell2mat(Data');

[pi, mu, Sigma] = Gmm(Xall, Q, 'diag');
% p_start0 = pi;
p_start0 = normalise(rand(Q,1));
for i1 = 1:Q
    Emis0{i1}.mu = mu(:,i1);
    Emis0{i1}.Sigma = Sigma(:,:,i1);
end

[p_start, A, Emis] = Chmm_gauss(Data, p_start0, A0, Emis0, 'cov_type', 'diag', 'cov_thresh', 1e-4)
for i1 = 1:length(Emis)
    Emis{i1}.mu
    Emis{i1}.Sigma
end

Ob = Gauss_p_xn_cond_zn(Data{1}, Emis);
[gamma, sum_ita, loglik] = ForwardBackward(p_start,A,Ob);
loglik;
path = ViterbiDecode(p_start, A, Ob);
path';


for i1 = 1:length(Emis)
    error_ellipse(Emis{i1}.Sigma, Emis{i1}.mu, 'style', 'r'); hold on
end
end

% =============================
function ChmmGaussMixTest()
rng(0)
X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/3, 10);
X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/3, 20);
X3 = mvnrnd([2,0], [0.5, 0; 0, 0.3]/3, 30);
X4 = mvnrnd([2,2], [0.5, 0.2; 0.2, 0.3]/3, 10);
X5 = mvnrnd([4,0], [0.3, -0.2; -0.2, 0.5]/3, 20);
X6 = mvnrnd([4,2], [0.5, 0; 0, 0.3]/3, 30);

X = [X1; X2; X3; X4; X5; X6];
Data{1} = [X1;X3;X5];
% Data{2} = [X2;X4;X6];
scatter(X(:,1), X(:,2), '.'); hold on

Q = 1; 
M = 3; 
p = 2; 

p_start0 = normalise(rand(Q,1));
for i1 = 1:Q-1
    A0(i1,i1:i1+1) = 0.5;
end
A0(Q,Q) = 1;

Xall = cell2mat(Data');

[pi, mu, Sigma] = Gmm(Xall, Q*M);
Emis0.pi = zeros(M,Q);
Emis0.mu = zeros(p,M,Q);
Emis0.Sigma = zeros(p,p,M,Q);
cnt = 1;
for i1 = 1:Q
    for i2 = 1:M
        Emis0.pi(i2,i1) = pi(cnt);
        Emis0.mu(:,i2,i1) = mu(:,cnt);
        Emis0.Sigma(:,:,i2,i1) = Sigma(:,:,cnt);
        cnt = cnt + 1;
    end
end

% ------------
[p_start, A, Emis] = Chmm_gauss_mix(Data, p_start0, A0, Emis0);
p_start
A
Emis

for i1 = 1:Q
    for i2 = 1:M
        error_ellipse(Emis.Sigma(:,:,i2,i1), Emis.mu(:,i2,i1), 'style', 'r'); hold on
    end
end


end

% ===========




