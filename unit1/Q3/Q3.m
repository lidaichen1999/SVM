
type='c'; 
gam = 1; 
disp('Linear kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

disp('Press any key to continue...'), pause, 

type='c'; 
gam = 1; 
t = 1; 
degree = 9;
disp('Polynomial kernel of degree 9'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
disp('Press any key to continue...'), pause,        

disp('RBF kernel')
%gam = 1; 
sig2list=[0.001,0.01, 0.1, 1, 10, 100,1000];
%sig2 = 0.1;
gamlist = [0.001,0.01, 0.1, 1, 10, 100,1000];
errlist=[];

for gam=gamlist,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end
