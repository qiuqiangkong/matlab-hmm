function path = ViterbiDecode(p_start, A, Ob)
[N,Q] = size(Ob);
path = zeros(N,1);
PATH = zeros(N,Q);

tmp = p_start'.*Ob(1,:);
tmp = normalise(tmp);
for i1 = 2:N
    C = bsxfun(@times, bsxfun(@times,tmp',A), Ob(i1,:));
    [tmp, points] = max(C,[],1);
    tmp = normalise(tmp);
    PATH(i1-1,:) = points';
end
[~,path(N)] = max(tmp);

for i1 = N-1:-1:1
    path(i1) = PATH(i1, path(i1+1));
end


end