% http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
function [X, Y, K_classes] = get_letterp2_data()
    
    fid = fopen('datasets/letter-recognition.data');
    t = textscan(fid,'%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d','delimiter',',');
    Y_categorical = t{1};
    fclose(fid);
    
    X = csvread('datasets/letter-recognition.data',0, 1);
    Y = zeros(size(Y_categorical));
    
    X(~isfinite(X)) = 0;
    
    positives = {'A','B','C','D','E','F','G','H','I','J','K','L','M'};
    
    for p = positives
       Y(strcmp(Y_categorical, p)) = 1;
    end
    
    Y(Y ~= 1) = 0;
    
    K_classes = 2;
end