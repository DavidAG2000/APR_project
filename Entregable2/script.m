
addpath("svm_apr");

%load "data/mini/trSep.dat"; load "data/mini/trSeplabels.dat"
load"data/mini/tr.dat"; load "data/mini/trlabels.dat"

res=svmtrain(xl,X,'-t 0 -c 1');

cLagrange = res.sv_coef
vector_soporte = res.SVs

vectorPesos=cLagrange' * vector_soporte
umbral=sign(res.sv_coef(1)) - vectorPesos*res.SVs(1,:)'

x1 = [0:7];
y1 = -vectorPesos(1)/vectorPesos(2)*x1 - umbral/vectorPesos(2);
y2 = -vectorPesos(1)/vectorPesos(2)*x1 - (umbral-1)/vectorPesos(2);
y3 = -vectorPesos(1)/vectorPesos(2)*x1 - (umbral+1)/vectorPesos(2);
margen = 2./sqrt(vectorPesos*vectorPesos')

disp(vectorPesos(1)/vectorPesos(2)),disp(umbral/vectorPesos(2))
toler = zeros(length(xl), 1);
toler_sv = zeros(length(res.sv_indices), 1);

for i = 1:length(res.sv_indices)
        toler(res.sv_indices(i)) = abs( (abs(res.sv_coef(i))==1000)*(1 - ( sign(res.sv_coef(i)) * (vectorPesos*X(res.sv_indices(i),:)' + umbral) )) );
        toler_sv(i) = toler(res.sv_indices(i));
end

plot(
    X(xl==1,1),X(xl==1,2),"sr","markerfacecolor","r",
    X(xl==2,1),X(xl==2,2),"ob","markerfacecolor","b",
    x1,y1,"-k",
    x1,y2,"-m",
    x1,y3,"-c",
    X(toler != 0, 1), X(toler != 0, 2), "xk", "markersize", 15,
    res.SVs(toler_sv == 0, 1), res.SVs(toler_sv == 0, 2), "+k", "markersize", 15
);
set(gca, "xgrid", "on");
set(gca, "ygrid", "on");
axis([0, 7, 0, 7],"square");

for i = 1:rows(X)
    if any(res.sv_indices == i)
        ind = find(res.sv_indices == i);
        text(X(res.sv_indices(ind),1)+0.15,X(res.sv_indices(ind),2),sprintf("%4.2f",toler(res.sv_indices(ind))),"FontSize",10);
        text(X(res.sv_indices(ind),1)-0.07,X(res.sv_indices(ind),2)+0.3,sprintf("%4.2f",abs(res.sv_coef(ind))),"FontSize",10);
    else
        text(X(i,1)+0.15,X(i,2),"0.00","FontSize",10);
        text(X(i,1)-0.07,X(i,2)+0.3,"0.00","FontSize",10);
    endif
endfor

for i = 1:rows(res.sv_indices)
    text(X(res.sv_indices(i),1)+0.15,X(res.sv_indices(i),2),sprintf("%4.2f",toler(res.sv_indices(i))),"FontSize",10);
        text(X(res.sv_indices(i),1)-0.07,X(res.sv_indices(i),2)+0.3,sprintf("%4.2f",abs(res.sv_coef(i))),"FontSize",10);
endfor

cadena=sprintf("margin = %.2f", margen);
text(0.25,6.35,cadena);

saveas (1, "no_sep_img.png")
