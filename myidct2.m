function y = myidct2(x,m,n,matrixform)


if nargin < 4
 matrixform = false;
end

y = idct2(reshape(x,m,n));

if ~matrixform
 y = y(:);
end
