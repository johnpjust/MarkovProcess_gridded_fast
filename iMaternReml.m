function [x, se] = iMaternReml(yin,win,initial,nseed)


if nargin == 3
 nseed = 50;
end

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

eve = rng;
rng(2441139);
RadVar = 2*(rand(n+q-1,nseed) < 0.5) - 1;
rng(eve);
options = optimset('Display','iter','TolFun',0.01,'TolX',0.001,'MaxFunEvals',500,'MaxIter',40);



x0 = [log(initial(2)); log( 2*initial(3)/(1-2*initial(3))); log(initial(4))];

gradiso = @(pars) gradfun(pars,initial(1),F,FF,Z,yield,Pr,Pc,xxr,xxc,xr,xc,r,c,n,q,RadVar,nseed);

[x, fval, exitflag, output, hess] = fsolve(gradiso,x0,options);

se = sqrt(diag(inv(-hess)));

lp = exp(x(1));
beta = 0.5/(1 + exp(-x(2)));
nu = exp(x(3));

se(1) = se(1)*lp;
se(2) = 0.5*se(2) * exp(x(2))/(1 + exp(x(2)))^2;
se(3) = se(3)*nu;

x = [lp;beta;nu];
% 
% ly = initial(1);
% 
% 
% 
% 
% L = lp*(4*beta*xxr + 4*(0.5-beta)*xxc).^nu;
% 
% W1a = (4*beta)^nu*(Pr*diag(xr.^nu)*Pr');
% W2a = (4*(0.5-beta))^nu*(Pc*diag(xc.^nu)*Pc');
% 
% thresh = quantile(abs( squareform(tril(W1a,-1))) , 1-9/r);
% W1a(abs(W1a) < thresh) = 0;
% W1a = sparse(W1a);
% W2a(abs(W2a) < thresh) = 0;
% W2a = sparse(W2a);
% 
% 
% C = ichol(ly*FF + lp*(kron(speye(c),W1a) + kron(W2a,speye(r))));
% Ct = C';
% 
% FFd = full(diag(FF));
% xqx = @(v) ly*(FFd.*v) + myidct2(L.*mydct2(v,r,c) , r,c);
% 
% 
% 
% M1 = @(vv) C\vv;
% M2 = @(vv) Ct\vv;
% 
% % Compute the BLUP
% 
% tic;
% [psi, flag, rel, iter] = symmlq(xqx,ly*Z,1e-12,q,M1,M2);
% toc
% 
% psi = reshape(psi,r,c);
% psi = psi(r:-1:1,:);

return




function grad = gradfun(pars,ly,F,FF,Z,yield,Pr,Pc,xxr,xxc,xr,xc,r,c,n,q,RadVar,nseed)

% using preconditioner.. assuming fixed nugget precision ly.

lp = exp(pars(1));
beta = 0.5/(1 + exp(-pars(2)));
nu = exp(pars(3));

lastelt = @(v) v(2:q);
Xt = @(vv) F'*vv(1:n) + myidct2([0;vv(n+1:n+q-1)],r,c) ;
X = @(vv) [F*vv; lastelt(mydct2(vv,r,c))];

Q = [ly*ones(n,1); lp*(4*beta*xxr(2:q) + 4*(0.5-beta)*xxc(2:q)).^nu ];

%dQ1 = [ones(n,1);zeros(q-1,1)];
dQ2 = [zeros(n,1);(4*beta*xxr(2:q) + 4*(0.5-beta)*xxc(2:q)).^nu];
dQ3 = [zeros(n,1); 4*lp*nu*(xxr(2:q) - xxc(2:q)).*(4*beta*xxr(2:q) + 4*(0.5-beta)*xxc(2:q)).^(nu-1)];
dQ4 = [zeros(n,1);Q(n+1:n+q-1).*log(4*beta*xxr(2:q) + 4*(0.5-beta)*xxc(2:q))];

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
[psi, flag, rel, iter] = symmlq(xqx,ly*Z,1e-12,q,M1,M2);
toc





g2 = 0;
g3 = 0;
g4 = 0;
% Computing the score function

parfor t=1:nseed
    u0 = RadVar(:,t);
    v0 = Xt(Q.*u0);
    [v,flag] = symmlq(xqx,v0,1e-12,q,M1,M2);
    v = u0 - X(v);
    %    g1 = g1 + sum(RadVar(:,t).*dQ1.*v./Q)/nseed;
    g2 = g2 + sum(u0.*dQ2.*v./Q)/nseed;
    g3 = g3 + sum(u0.*dQ3.*v./Q)/nseed;
    g4 = g4 + sum(u0.*dQ4.*v./Q)/nseed;
    
end

res2 = ( [yield; zeros(q-1,1)] - X(psi) ).^2;
%g1 = g1 - sum(res2.*dQ1);
g2 = g2 - sum(res2.*dQ2);
g3 = g3 - sum(res2.*dQ3);
g4 = g4 - sum(res2.*dQ4);

grad = [g2;g3;g4];

grad = 0.5*grad.*[lp;0.5*exp(pars(2))/(1 + exp(pars(2)))^2;nu];

disp([lp beta nu grad']);

return
