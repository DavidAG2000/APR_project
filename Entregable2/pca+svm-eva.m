#!/usr/bin/octave -qf

if (nargin != 8)
    printf("Usage: pca+svm-eva.m <trdata> <trlabels> <tedata> <telabs> <pcaK> <ts> <cs> <gs>\n")
    exit(1);
end;

%./pca+svm-eva.m train-images-idx3-ubyte.mat.gz train-labels-idx1-ubyte.mat.gz t10k-images-idx3-ubyte.mat.gz t10k-labels-idx1-ubyte.mat.gz "[100]" "[2]" "[10]"  "[1.0e-02]"
addpath("svm_apr");

arg_list = argv();
trdata = arg_list{1};
trlabs = arg_list{2};
tedata = arg_list{3};
telabs = arg_list{4};
pcaKs = str2num(arg_list{5});
ts = str2num(arg_list{6});
cs = str2num(arg_list{7});
gs = str2num(arg_list{8});

load(trdata);
load(trlabs);
load(tedata);
load(telabs);
X = X/255;
Y = Y/255;
%PCA
[m, w] = pca(X);
Xtr = X - m;
Xdv = Y - m;
N = rows(Y);


for k = 1:length(pcaKs)
    proyX = Xtr * w(:, 1:pcaKs(k));
    proyY = Xdv * w(:, 1:pcaKs(k));
    for c=1:length(cs)
      for j=1:length(ts)
        for i=1:length(gs)
            printf("Entrenando \n");
            printf(" %d \t %d \t %d \t %d \n",pcaKs(k),ts(j),cs(c),gs(i));
            res = svmtrain(xl, proyX, ["-q -t ", num2str(ts(j)), " -c ", num2str(cs(c)), " -g ", num2str(gs(i))]);
            printf("Prediciendo \n");
            [pred, accuracy, d] = svmpredict(yl, proyY, res, '-q');
            p = accuracy(1) / 100;
            intervalo = 1.96* sqrt((p * (1-p))/N);
            printf("T \t C \t G \t acierto \t intervalo \n");
            printf("%d \t %d \t %d \t %6.3f \t [%.3f  %.3f] \n",ts(j),cs(c),gs(i),p * 100,(p-intervalo)*100,(p+intervalo)*100);
        end
      end
    end
end