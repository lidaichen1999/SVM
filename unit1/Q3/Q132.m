%perf1 = rsplitvalidate({Xtrain, Ytrain,'c',gam,sig2,'RBF_kernel'},0.80,'misclass');% split training and validation
%perf2 = crossvalidate({Xtrain, Ytrain,'c',gam,sig2,'RBF_kernel'},10,'misclass');
%perf3 = leaveoneout({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},'misclass');

perf1list = [];
for gam = gamlist
    for sig2 = sig2list
        perf1 = rsplitvalidate({Xtrain, Ytrain,'c',gam,sig2,'RBF_kernel'},0.80,'misclass');
        perf1list(end+1,:) = [gam, sig2, perf1];
    end
end
figure;
X = reshape(perf1list(:,1), [], length(sig2list));
Y = reshape(perf1list(:,2), [], length(gamlist));
Z = reshape(perf1list(:,3), [], length(sig2list));
%surf(X, Y, Z);
h = heatmap(gamlist,sig2list,Z);
h.xlabel('gam');
h.ylabel('sig2');
h.title('random split performance');

perf2list = [];
for gam = gamlist
    for sig2 = sig2list
        perf2 = crossvalidate({Xtrain, Ytrain,'c',gam,sig2,'RBF_kernel'},10,'misclass');
        perf2list(end+1,:) = [gam, sig2, perf2];
    end
end
figure;
%X = reshape(perf1list(:,1), [], length(sig2list));
%Y = reshape(perf1list(:,2), [], length(gamlist));
Z = reshape(perf2list(:,3), [], length(sig2list));
%surf(X, Y, Z);
h = heatmap(gamlist,sig2list,Z);
h.xlabel('gam');
h.ylabel('sig2');
h.title('crossvalidation performance');

perf3list = [];
for gam = gamlist
    for sig2 = sig2list
        perf3 = leaveoneout({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},'misclass');
        perf3list(end+1,:) = [gam, sig2, perf3];
    end
end
figure;
%X = reshape(perf1list(:,1), [], length(sig2list));
%Y = reshape(perf1list(:,2), [], length(gamlist));
Z = reshape(perf3list(:,3), [], length(sig2list));
%surf(X, Y, Z);
h = heatmap(gamlist,sig2list,Z);
h.xlabel('gam');
h.ylabel('sig2');
h.title('leave-one-out performance');

[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'misclass'});