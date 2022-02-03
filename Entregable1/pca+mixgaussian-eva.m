#!/usr/bin/octave -qf

if (nargin != 7)
    printf("Usage: pca+mixgaussian-eva.m <trdata> <trlabels> <tedata> <telabs> <pcaK> <Ks> <alphas>\n")
    exit(1);
end;

arg_list = argv();
trdata = arg_list{1};
trlabs = arg_list{2};
tedata = arg_list{3};
telabs = arg_list{4};
pcaK = str2num(arg_list{5});
K = str2num(arg_list{6});
alpha = str2num(arg_list{7});


load(trdata);
load(trlabs);
load(tedata);
load(telabs);

printf("\n  alpha  pca  Ks  dv-err");
printf("\n-------  ---  --- ------\n");

[m, w] = pca(X);
Xtr = X - m;
Xdv = Y - m;

proyX = Xtr * w(:, 1:pcaK);
proyY = Xdv * w(:, 1:pcaK);

edv = mixgaussian(proyX, xl, proyY, yl, K, alpha);
printf("%.1e %3d %3d %6.3f\n\n\n", alpha, pcaK, K, edv);

m = edv / 100;
s = sqrt(m * (1 - m) / rows(Xdv));
interval = 1.96 * s;
printf(" Error:%.4d \t Intervalo de confianza [%.4d , %4d]\n", edv, edv - interval, edv + interval);
