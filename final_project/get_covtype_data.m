% http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
function [X, Y, K_classes] = get_covtype_data()
    if exist('datasets/covtypeX.mat') && exist('datasets/covtypeY.mat')
        load('datasets/covtypeX');
        load('datasets/covtypeY');
        X = covtypeX;
        Y = covtypeY;
        K_classes = 2;
        return;
    end
    
    d = csvread('datasets/covtype.data');
    Y = d(:,end);
    X = d(:,1:end-1);
    
    X(~isfinite(X)) = 0;
    
    lodgepole_pine_code_type_designation = 2;
    Y((Y ~= lodgepole_pine_code_type_designation)) = 0;
    Y((Y == lodgepole_pine_code_type_designation)) = 1;
    
    K_classes = 2;
    
    covtypeX = X;
    covtypeY = Y;
    save('datasets/covtypeX', 'covtypeX');
    save('datasets/covtypeY', 'covtypeY');
end
