function psi = iMaternReml_predict(yin,win,x)


[r, c] = size(yin);


y = yin(r:-1:1,:);
w = win(r:-1:1,:);


q = r*c;

yield = y(w>0);
n = length(yield);

idx = find(w>0);
F = sparse(1:n,idx,1,n,q);

xr = sin(0.5*pi*(0:(r-1))'/r).^2;
xc = sin(0.5*pi*(0:(c-1))'/c).^2;

xxr = kron(ones(c,1),xr);
xxc = kron(xc,ones(r,1));

Pr = [ones(r,1)/sqrt(r), sqrt(2/r) * cos( ((1:r)-0.5)'*(1:(r-1)) * pi/r )];
Pc = [ones(c,1)/sqrt(c), sqrt(2/c) * cos( ((1:c)-0.5)'*(1:(c-1)) * pi/c )];


FF = F'*F;
Z = F'*yield;

% eve = rng;
% rng(2441139);
% rng(eve);





ly = 10000;


lp = x(1);
beta = x(2);
nu = x(3);

L = lp*(4*beta*xxr + 4*(0.5-beta)*xxc).^nu;

W1a = (4*beta)^nu*(Pr*diag(xr.^nu)*Pr');
W2a = (4*(0.5-beta))^nu*(Pc*diag(xc.^nu)*Pc');

thresh = quantile(abs( squareform(tril(W1a,-1))) , 1-9/r);
W1a(abs(W1a) < thresh) = 0;
W1a = sparse(W1a);
W2a(abs(W2a) < thresh) = 0;
W2a = sparse(W2a);


C = ichol(ly*FF + lp*(kron(speye(c),W1a) + kron(W2a,speye(r))));
Ct = C';

FFd = full(diag(FF));
xqx = @(v) ly*(FFd.*v) + myidct2(L.*mydct2(v,r,c) , r,c);



M1 = @(vv) C\vv;
M2 = @(vv) Ct\vv;

% Compute the BLUP

tic;
[psi, ~, ~, ~] = symmlq(xqx,ly*Z,1e-12,q,M1,M2);
toc

psi = reshape(psi,r,c);
psi = psi(r:-1:1,:);

return
