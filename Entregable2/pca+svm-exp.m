#!/usr/bin/octave -qf

if (nargin != 9)
  printf("Usage: pca+svm-exp.m <trdata> <trlabels> <pcaKs> <cs> <ts> <ds> <gs> <%%trper> <%%dvper>\n")
  exit(1);
end;
%./pca+svm-exp.m train-images-idx3-ubyte.mat.gz train-labels-idx1-ubyte.mat.gz "[50 100 200]" "[1 10 100]" "[0 1 2 3]" "[1 2 3 4 5]" "[1.0e-01 1.0e-02 1.0e-03 1.0e-04 1.0e-05]" 9 1

addpath("svm_apr");

arg_list = argv();
trdata = arg_list{1};
trlabs = arg_list{2};
pcaKs = str2num(arg_list{3});
cs = str2num(arg_list{4});
ts = str2num(arg_list{5});
ds = str2num(arg_list{6});
gs = str2num(arg_list{7});
trper = str2num(arg_list{8});
dvper = str2num(arg_list{9});


load(trdata);
load(trlabs);

N = rows(X);
seed = 23; 
rand("seed", seed); 
permutation = randperm(N);
X = X(permutation, :);
X = X/255; 
xl = xl(permutation, :);
Ntr = round(trper / 100 * N);
Ndv = round(dvper / 100 * N);
Xtr = X(1:Ntr, :);
xltr = xl(1:Ntr);
Xdv = X(N - Ndv + 1:N, :);
xldv = xl(N - Ndv + 1:N);

[m W] = pca(Xtr);
%Los vectores de proyección en W están por columnas
Xtr = Xtr - m;
Xdv = Xdv - m;

filename = "pca+svm-exp.out";
fileRes = fopen (filename, "w");

for k  = 1:length(pcaKs)
  pcaXtr = Xtr * W(:, 1:pcaKs(k));
  pcaXdv = Xdv * W(:, 1:pcaKs(k));
  fprintf(fileRes,"PCA= %d \n",pcaKs(k))
  for c=1:length(cs)
      for j=1:length(ts)
          if ts(j) == 1
            fprintf(fileRes,"T \t C \t D \t acierto \t intervalo \n");
                for i=1:length(ds)
                    res = svmtrain(xltr, pcaXtr, ["-q -t ", num2str(ts(j)), " -c ", num2str(cs(c)), " -d ", num2str(ds(i))]);
                    [pred, accuracy, d] = svmpredict(xldv, pcaXdv, res, '-q');
                    p = accuracy(1) / 100;
                    intervalo = 1.96* sqrt((p * (1-p))/N);
                    fprintf(fileRes,"%d \t %d \t %d \t %3f \t %3f   \n",ts(j),cs(c),ds(i),p,intervalo);
                end
          elseif ts(j) == 0
                fprintf(fileRes,"T \t C \t acierto \t intervalo \n");
                res = svmtrain(xltr, pcaXtr, ["-q -t ", num2str(ts(j)), " -c ", num2str(cs(c))]);
                [pred, accuracy, d] = svmpredict(xldv, pcaXdv, res, '-q');
                p = accuracy(1) / 100;
                intervalo = 1.96* sqrt((p * (1-p))/N);
                fprintf(fileRes,"%d \t %d \t %3f \t %3f   \n",ts(j),cs(c),p,intervalo);
          else
                fprintf(fileRes,"T \t C \t G \t acierto \t intervalo \n");
                for i=1:length(gs)
                    res = svmtrain(xltr, pcaXtr, ["-q -t ", num2str(ts(j)), " -c ", num2str(cs(c)), " -g ", num2str(gs(i))]);
                    [pred, accuracy, d] = svmpredict(xldv, pcaXdv, res, '-q');
                    p = accuracy(1) / 100;
                    intervalo = 1.96* sqrt((p * (1-p))/N);
                    fprintf(fileRes,"%d \t %d \t %d \t %3f \t %3f   \n",ts(j),cs(c),gs(i),p,intervalo);
                end
          endif
          fprintf(fileRes,"\n");
      end
  end
end
printf("fichero completado");
fclose (fileRes);
