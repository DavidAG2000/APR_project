#!/usr/bin/octave -qf
if(nargin != 6)
  printf("Usage: mixgaussian-exp.m <trdata> <trlabels> <Ks> \
  <alphas> <%%trper> <%%dvper>\n");
  exit(1);
end;

arg_list = argv();
trdata = arg_list{1};
trlabs = arg_list{2};
Ks = str2num(arg_list{3});
alphas = str2num(arg_list{4});
trper = str2num(arg_list{5});
dvper = str2num(arg_list{6});

load(trdata);
load(trlabs);

N = rows(X);
seed = 23;
rand("seed", seed);
permutation = randperm(N);
X = X(permutation, :);
xl = xl(permutation, :);

Ntr = round(trper / 100 * N);
Ndv = round(dvper / 100 * N);
Xtr = X(1:Ntr, :);
xltr = xl(1:Ntr);
Xdv = X(N - Ndv + 1:N, :);
xldv = xl(N - Ndv + 1:N);

printf("\n alpha K dv-err");
printf("\n------- --- ------\n");
for i = 1:length(alphas)
for k = 1:length(Ks)
  edv = mixgaussian(Xtr, xltr, Xdv, xldv, Ks(k), alphas(i));
  printf("%.1e %3d %6.3f\n", alphas(i), Ks(k), edv);
end
end