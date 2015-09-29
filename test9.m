function test9
X = zeros(8,5);
a = [2 1 3 5 4 3 3 2 ];
b = sub2ind(size(X), 1:8, a);
X(b) = 1;