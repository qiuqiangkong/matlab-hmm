
function Ob = Gauss_xn_cond_zn(X, Emis)
N = size(X,1);
Q = length(Emis);
Ob = zeros(N,Q);
for i1 = 1:Q
    Ob(:,i1) = mvnpdf(X, Emis{i1}.mu', Emis{i1}.Sigma);
%     Emis{i1}.mu'
%     Emis{i1}.Sigma
end
% mvnpdf(X, Emis{3}.mu', Emis{3}.Sigma)
% pause

end