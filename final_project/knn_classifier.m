function [Y_predict] = knn_classifier(k, X_train, Y_train,...
                                      X_test, weighting_policy_enum,...
                                      kernel_width)
                                  
    assert(0 <= weighting_policy_enum <= 4, 'invalid weighting policy');
    
    m = size(X_test, 1); % m = # of houses
    n = size(X_test, 2); % n = # of attributes/features
    
    Y_predict = zeros(m, 1);
    
    neighborhood = X_train;
    houses = X_test;
    
    % let  m = | { houses } |
    % and  k = | { nearest neighbors indices } |
    % then dimensions(k_nearest_neighbor_indices_foreach_house) = m by k
    % and  dimensions(k_nearest_neighbor_distances_foreach_house) = m by k
    if weighting_policy_enum ~= 4
        nonweighted = ones(1, n);
    
        [k_nearest_neighbor_indices_foreach_house,...
         k_nearest_neighbor_distances_foreach_house] =...
                    knnsearch(houses, neighborhood, k, nonweighted);
    else
        % perform information gain ratio weighting
        gain_ratios = calculate_information_gain_ratios(X_train, Y_train);
        
        [k_nearest_neighbor_indices_foreach_house,...
         k_nearest_neighbor_distances_foreach_house] =...
                    knnsearch(houses, neighborhood, k, gain_ratios);
    end
            
    for i = 1:m
       %types are enums (ie integers representing categorical-nominal data)
       neighbors_house_types =...
           Y_train(k_nearest_neighbor_indices_foreach_house(i,:));
       
       neighbors_house_distances =...
            k_nearest_neighbor_distances_foreach_house(i,:);
       
       Y_predict(i) = calculate_house_type(neighbors_house_types,...
                                           neighbors_house_distances,...
                                           weighting_policy_enum,...
                                           kernel_width);
    end
end

function [house_type] = calculate_house_type(neighbors_house_types,...
                                             neighbors_house_distances,...
                                             weighting_policy_enum,...
                                             kernel_width)
                                       
    k = length(neighbors_house_types); % k = number of neighbors
                                         
    switch weighting_policy_enum
        % euclidian distance
        case 1
            house_type = (1/k)*sum(neighbors_house_types);
        
        % distance-weighted
        case 2
            weights = 1 ./ neighbors_house_distances;
            house_type = sum(weights'.*neighbors_house_types)/sum(weights);
        
        % locally-weighted averaging
        case 3
            weights = 1 ./ exp(kernel_width*neighbors_house_distances);
            house_type = sum(weights'.*neighbors_house_types)/sum(weights);
            
        % gain-ratio weighted
        case 4
            house_type = (1/k)*sum(neighbors_house_types);
            % weighting with gain ratios has been taken care of in the
            % caller function
            
        % just calculate euclidian distance
        otherwise
            house_type = (1/k)*sum(neighbors_house_types);
    end
    
    if ~isfinite(house_type)
        house_type = 0;
    else
        house_type = round(house_type);
    end
end

function [gain_ratios] = calculate_information_gain_ratios(X, Y)
    m_examples = size(X,1);
    n_attributes = size(X,2); 
    gain_ratios = zeros(1,n_attributes);
    S = X;
    
    entropy_S = calculate_entropy_for_binary_classification(Y);
    size_S = m_examples;
    
    for A = 1:n_attributes
       % subset of S with value A = 0
       idx = find(S(:,A) == 0);
       S_0 = S(idx,:);
       entropy_S_0 = calculate_entropy_for_binary_classification(Y(idx));
       size_S_0 = size(S_0, 1);

       % subset of S with value A = 1
       idx = find(S(:,A) == 1);
       S_1 = S(idx,:);
       entropy_S_1 = calculate_entropy_for_binary_classification(Y(idx));
       size_S_1 = size(S_1, 1);

       gain_ratios(A) = entropy_S - (size_S_0/size_S)*entropy_S_0...
                                  - (size_S_1/size_S)*entropy_S_1;
    end
end

function [entropy] = calculate_entropy_for_binary_classification(binary_labels)
    prob_pos = sum(binary_labels)/length(binary_labels);
    prob_neg = 1 - prob_pos;
    entropy = -(prob_pos)*log2(prob_pos) - (prob_neg)*log2(prob_neg);
end