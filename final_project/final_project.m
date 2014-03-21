% UCSD COGS 118A Natural Computation I Final Project
%
% Simulate learning algorithms comparison experiment as described in:
% http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf
%
% Kevin Hung 3/15/2014
function [] = final_project()
    % add folder containing machine learning algorithm src codes
    addpath(genpath('lib'));
    
    K_FOLD_CROSSVALIDATION = 5;

    classifiers = {'knn','svm', 'rf', 'adaboost', 'ann'};
    datasets = {'letterp2', 'adult', 'covtype'};
    test_sizes = [14000, 35222, 25000];
    training_size = 5000;
    
    classifier_dataset_prediction_labels_table =...
            cell(length(classifiers), length(datasets));
    
    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};

        switch dataset_name
            case 'letterp2'; [X, Y, ~] = get_letterp2_data();
            case 'adult'; [X, Y, ~] = get_adult_data();
            case 'covtype'; [X, Y, ~] = get_covtype_data();
            otherwise; break
        end
        
        fprintf('\nFinding model for %s.data...\n',datasets{dataset_idx});
        
        m_examples = size(X,1);
        rng = RandStream('mt19937ar','Seed',1);
        training_indices = randperm(rng, m_examples, training_size);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% CROSS-VALIDATION                                              %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Perform cross-validation on the training set with classifiers,
        % based on simple explanation of k-fold crossvalidation method:
        % http://statweb.stanford.edu/~tibs/sta306b/cvwrong.pdf
        num_trials = K_FOLD_CROSSVALIDATION;
        
        % K-Nearest Neighbor Binary Classification Cross-Validation Procedure
        % http://www.mathworks.com/matlabcentral/fileexchange/19345-efficient-k-nearest-neighbor-search-using-jit
        knn_number_of_parameter_settings = 14;
        knn_parameter_errors = zeros(num_trials,... % find least error
                                     knn_number_of_parameter_settings);      
        K_neighbors = 26;
        knn_weighting_policies = {'euclidian_dist','distance_weighted',...
                                 'locally_weighted','gain_ratio_weighted'};
                             
        % RF Classification Cross-Validation Procedure
        % http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
        N_TREES = 1024;
        FEATURE_SIZE_SPLITS = [1,2,4,6,8,12,16,20];
        rf_number_of_parameter_settings = numel(FEATURE_SIZE_SPLITS);
        rf_parameter_errors = zeros(num_trials,...
                                    rf_number_of_parameter_settings);
                                
        % Adaboost Classification Cross-Validation Procedure
        % http://www.mathworks.com/matlabcentral/fileexchange/21317-adaboost
        log2_steps = 1:3;
        adaboost_number_of_parameter_settings = numel(log2_steps);
        adaboost_parameter_errors = zeros(num_trials,...
                                    adaboost_number_of_parameter_settings);
        
        % Artificial Neural Network Binary Cross-Validation Procedure
        % http://www.mathworks.com/products/neural-network/code-examples.html
        hidden_units_sizes = 2.^(0:7);
        momentum_range = [0, 0.2, 0.5, 0.9];
        [hidden_units, momentums] = meshgrid(hidden_units_sizes, momentum_range);
        ann_number_of_parameter_settings = numel(hidden_units);
        ann_parameter_errors = zeros(num_trials,...
                                    ann_number_of_parameter_settings);
                                
        first_example_on_the_border = 1;
        
        for k = 1:K_FOLD_CROSSVALIDATION
            validation_size = training_size / num_trials;
            last_example_on_the_border = k * validation_size;
            
            validation_indices = training_indices(:,...
                   first_example_on_the_border:last_example_on_the_border);
               
            control_indices =...
                setdiff(training_indices, validation_indices);
            
            m_cv = validation_size;
            X_cv = X(validation_indices,:);
            Y_cv = Y(validation_indices,:);
            X_control = X(control_indices,:);
            Y_control = Y(control_indices,:);
            
            for classifier_idx = 1:numel(classifiers)
                classifier = classifiers{classifier_idx};
                
                switch classifier
                    
                    % K-Nearest Neighbor Binary Classification
                    case 'knn'
                        fprintf('\n\tCross-validation KNN model, trial %d...\n',k);
            
                        for wp_idx = 1:numel(knn_weighting_policies)
                            
                            fprintf('\t\tFitting KNN %s model...\n',knn_weighting_policies{wp_idx});
                            
                            if wp_idx == 3 % 'locally_weighted'
                                for log_2_kernel_width = 0:10
                                   kernel_width = 2^log_2_kernel_width; 
                                   Y_predict=knn_classifier(K_neighbors,...
                                               X_control, Y_control,...
                                               X_cv, wp_idx, kernel_width);
                                           
                                   cv_error = sum((Y_predict-Y_cv).^2)/(2*m_cv);
                                   knn_parameter_errors(k,wp_idx+log_2_kernel_width) = cv_error;
                                end
                            else
                                Y_predict = knn_classifier(K_neighbors,...
                                                  X_control, Y_control,...
                                                  X_cv, wp_idx, 0);

                                cv_error=sum((Y_predict-Y_cv).^2)/(2*m_cv);
                                
                                if wp_idx == 4
                                    knn_parameter_errors(k, 14) = cv_error;
                                    continue;
                                end
                                
                                knn_parameter_errors(k, wp_idx) = cv_error;
                            end
                        end
                        
                    % SVM Binary Classification
                    case 'svm'
                        continue;
                        
                        
                    % RF Binary Classification
                    case 'rf'
                        fprintf('\n\tCross-validation RF model, trial %d...\n',k);
                        
                        for split_idx = 1:numel(FEATURE_SIZE_SPLITS)
                            split = FEATURE_SIZE_SPLITS(split_idx);
                            fprintf('\t\tFitting RF model with split = %d...\n',split);
                            
                            % http://vision.ucsd.edu/~pdollar/toolbox/doc/classify/forestTrain.html
                            pTrain={'F1',split,'M',N_TREES};
                            forest=forestTrain(X_control, Y_control+1, pTrain{:});
                            Y_predict = forestApply(single(X_cv),forest) - 1;
                            cv_error = sum((Y_predict-Y_cv).^2)/(2*m_cv);
                            rf_parameter_errors(k, split_idx) = cv_error;
                        end
                        
                    % Adaboost Binary Classification
                    case 'adaboost'
                        fprintf('\n\tCross-validation Adaboost model, trial %d...\n',k);
                        total_number_of_iterations = max(2.^log2_steps);
                        
                        for i=1:total_number_of_iterations
                            fprintf('\t\tFitting Adaboost model step=%d...\n',i);
                            
                            adaboost_model = ADABOOST_tr(@threshold_tr,@threshold_te,X_control,Y_control,i);
                            [Y_likelihoods,~] = ADABOOST_te(adaboost_model,@threshold_te,X_cv,Y_cv);
                            Y_predict = likelihood2class(Y_likelihoods)-1;
                            
                            if (bitand(i, i-1) == 0) % check if power of 2
                                if (min(log2_steps) <= log2(i) && log2(i) <= max(log2_steps))
                                    cv_error = sum((Y_predict-Y_cv).^2)/(2*m_cv);
                                    adaboost_parameter_errors(k, log2(i)) = cv_error;
                                end
                            end
                        end
                        
                        
                    % Artificial Neural Network Binary Classification                
                    case 'ann'
                        fprintf('\n\tCross-Validation Artificial Neural Network model, trial %d...\n',k);
                        setdemorandstream(491218382) % set seed to random symmetry breaking

                        x_tr = X_control';
                        y_tr = [1 - Y_control' ; Y_control'];
                        x_cv = X_cv';

                        for i = 1:ann_number_of_parameter_settings
                            n_hidden_units_opt = hidden_units(i);
                            momentum = momentums(i);
                            fprintf('\t\tHyperparameter %d (hidden_units, momentum) = (%d, %d)...\n',i, n_hidden_units_opt, momentum);

                            net = patternnet(n_hidden_units_opt);
                            net.trainFcn = 'traingdm';   % use gradient descent with momentum
                            net.trainParam.mc = momentum;
                            net.trainParam.showWindow = 0;
                            net.divideParam.trainRatio = 1; % use my own train,cv, and test set
                            net.divideParam.valRatio = 0;
                            net.divideParam.testRatio = 0;
                            
                            [net,~] = train(net,x_tr,y_tr);

                            y_prob = net(x_cv);
                            y_pred = vec2ind(y_prob) - 1;

                            Y_predict = y_pred';
                            cv_error = sum((Y_predict-Y_cv).^2)/(2*m_cv);
                            ann_parameter_errors(k, i) = cv_error;
                        end
                        
                    otherwise; break
                end
            end
            
            first_example_on_the_border = last_example_on_the_border + 1;
        end
        
        if (ismember('svm',unique(classifiers)))
            % SVM Binary Classification Cross-Validation Procedure
            t0 = tic;
            fprintf('\n\tCross-validation SVM model...\n');

            CLASSIFICATION = 0;
            LINEAR_KERNEL = 0;
            POLYNOMIAL_KERNEL = 1;
            RADIAL_KERNEL = 2;
            log10_C = -2:1;
            radial_widths = [0.001,0.005,0.01,0.05,0.1,0.5,1,2];

            % foreach C value, [1 linear, 2 polynomial, 7 radial]
            svm_number_of_parameter_settings = 10;
            svm_parameter_accuracies =...
                zeros(length(log10_C), svm_number_of_parameter_settings);

            fprintf('\t\tFitting SVM linear, polynomial{2,3}, and rbf models...\n');
            for i = 1:numel(log10_C)
               c = 10^(log10_C(i));
               fprintf('\t\tFitting SVM model with C = %d...\n', c);

               common_option = sprintf('-s %d -c %d -q', CLASSIFICATION , c);
               with_kfold_cv = sprintf('-v %d', K_FOLD_CROSSVALIDATION);

               linear_option = sprintf('-t %d', LINEAR_KERNEL);
               linear_kernel_accuracy = svmtrain(Y_train, X_train,...
                          [common_option ' ' linear_option ' ' with_kfold_cv]);
               svm_parameter_accuracies(i, 1) = linear_kernel_accuracy;

               polynomial_degree2_option = sprintf('-t %d -d %d', POLYNOMIAL_KERNEL, 2);
               polynomial_degree2_accuracy = svmtrain(Y_train, X_train,...
                    [common_option ' ' polynomial_degree2_option ' ' with_kfold_cv]);
                svm_parameter_accuracies(i, 2) = polynomial_degree2_accuracy;

               polynomial_degree3_option = sprintf('-t %d -d %d', POLYNOMIAL_KERNEL, 3);
               polynomial_degree3_accuracy = svmtrain(Y_train, X_train,...
                    [common_option ' ' polynomial_degree3_option ' ' with_kfold_cv]);
                svm_parameter_accuracies(i, 3) = polynomial_degree3_accuracy;

                for ii = 1:numel(radial_widths)
                   r = radial_widths(ii);
                   radial_option = sprintf('-t %d -g %d', RADIAL_KERNEL, r);
                   radial_accuracy = svmtrain(Y_train, X_train,...
                       [common_option ' ' radial_option ' ' with_kfold_cv]);
                   svm_parameter_accuracies(i, 3+ii) = radial_accuracy;
                end
            end

            tf = toc(t0);
            fprintf('\n\tFinished Cross-validating SVM model in %f secs\n',tf);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%                    TESTING                        %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Perform testing using optimal classifier models
        nontraining_indices = setdiff(1:m_examples, training_indices);
        testing_indices = nontraining_indices(...
                                randperm(rng,m_examples - training_size,...
                                    test_sizes(dataset_idx)));
                                
        X_train = X(training_indices,:);
        Y_train = Y(training_indices,:);
         
        X_test = X(testing_indices,:);
        Y_test = Y(testing_indices,:);
        
        for classifier_idx = 1:numel(classifiers)
        classifier = classifiers{classifier_idx};

            switch classifier
                case 'knn'
                    % KNN Optimal Parameter Testing Procedure
                    fprintf('\nTesting optimal KNN model...\n'); 
                    
                    [~, knn_optimal_parameter_idx] = min(knn_parameter_errors(:));
                    [~, parameter_settings_idx] = ind2sub(size(knn_parameter_errors),knn_optimal_parameter_idx);

                    if (parameter_settings_idx <= 2)
                        Y_predict_knn = knn_classifier(K_neighbors, X_train,...
                                                   Y_train, X_test,...
                                                   knn_optimal_parameter_idx, 0);
                    elseif (parameter_settings_idx == 14)
                        Y_predict_knn = knn_classifier(K_neighbors, X_train,...
                                                   Y_train, X_test, 4, 0);
                    else
                        Y_predict_knn = knn_classifier(K_neighbors, X_train,...
                                                   Y_train, X_test, 3,...
                                                   2^(knn_optimal_parameter_idx - 3));
                    end
                    
                    locally_weighted_distance_opt = 0;
                    
                    if parameter_settings_idx ~= 14
                        policy_opt = knn_weighting_policies{parameter_settings_idx};
                    else
                        policy_opt = 'locally_weighted';
                        locally_weighted_distance_opt = 2^(parameter_settings_idx-3);
                    end
                        
                    Y_test_knn = Y_test;
                    classifier_dataset_prediction_labels_table{1, dataset_idx} = Y_predict_knn;
                    save(['target/' dataset_name '_' classifier], 'Y_predict_knn', 'Y_test_knn', 'policy_opt', 'locally_weighted_distance_opt');

                case 'svm'
                    % SVM Optimal Parameter Testing Procedure
                    fprintf('\nTesting optimal svm model...\n'); 
                    
                    [~, svm_optimal_parameter_idx] = max(svm_parameter_accuracies(:));
                    [log10_C_opt, parameter_settings_idx] = ind2sub(size(svm_parameter_accuracies),svm_optimal_parameter_idx);

                    common_option = sprintf('-s %d -c %d -q', CLASSIFICATION , 10^log10_C_opt);
                    radius_opt = 0;
                    kernel = '';
                    
                    if (parameter_settings_idx == 1)
                        linear_kernel_model = svmtrain(Y_train, X_train, [common_option ' ' linear_option]);
                        Y_predict_svm = svmpredict(Y_test, X_test, linear_kernel_model);
                        kernel = 'linear';
                    elseif (parameter_settings_idx == 2)
                        polynomial_degree2_model = svmtrain(Y_train, X_train, [common_option ' ' polynomial_degree2_option]);
                        Y_predict_svm = svmpredict(Y_test, X_test, polynomial_degree2_model);
                        kernel = 'poly2';
                    elseif (parameter_settings_idx == 3)
                        polynomial_degree3_model = svmtrain(Y_train, X_train, [common_option ' ' polynomial_degree3_option]);
                        Y_predict_svm = svmpredict(Y_test, X_test, polynomial_degree3_model);
                        kernel = 'poly3';
                    else
                        radius_opt = radial_widths(parameter_settings_idx-3);
                        radial_option =sprintf('-t %d -g %d',RADIAL_KERNEL,radius_opt);
                        rbf_model = svmtrain(Y_train, X_train, [common_option ' ' radial_option]);
                        Y_predict_svm = svmpredict(Y_test, X_test, rbf_model);
                        kernel = 'radial';
                    end

                    Y_test_svm = Y_test;
                    classifier_dataset_prediction_labels_table{2, dataset_idx} = Y_predict_svm;
                    save(['target/' dataset_name '_' classifier], 'Y_predict_svm', 'Y_test_svm', 'radius_opt', 'kernel');

                case 'rf'
                    % RF Optimal Parameter Testing Procedure
                    fprintf('\nTesting optimal random forest model...\n');
                    
                    [~, rf_optimal_parameter_idx] = min(rf_parameter_errors(:));
                    [~, parameter_settings_idx] = ind2sub(size(rf_parameter_errors), rf_optimal_parameter_idx);
                    
                    feature_split_size_opt = FEATURE_SIZE_SPLITS(parameter_settings_idx);
                    pTrain={'F1', feature_split_size_opt,'M', N_TREES};
                    forest=forestTrain(X_train, Y_train+1, pTrain{:});
                    Y_predict_rf = forestApply(single(X_test), forest) - 1;
                    Y_test_rf = Y_test;
                    
                    classifier_dataset_prediction_labels_table{3, dataset_idx} = Y_predict_rf;
                    save(['target/' dataset_name '_' classifier], 'Y_predict_rf', 'Y_test_rf', 'feature_split_size_opt');

                case 'adaboost'
                    % Adaboost Optimal Parameter Testing Procedure
                    fprintf('\nTesting optimal adaboost model...\n');
                    
                    [~, adaboost_optimal_parameter_idx] = min(adaboost_parameter_errors(:));
                    [~, parameter_settings_idx] = ind2sub(size(adaboost_parameter_errors),adaboost_optimal_parameter_idx);
                    
                    n_steps_opt = 2^parameter_settings_idx;       
                    adaboost_model = ADABOOST_tr(@threshold_tr,@threshold_te,X_train,Y_train,n_steps_opt);
                    [Y_likelihoods,~] = ADABOOST_te(adaboost_model,@threshold_te,X_test,Y_test);
                    Y_predict_adaboost = likelihood2class(Y_likelihoods)-1;
                    Y_test_adaboost = Y_test;

                    classifier_dataset_prediction_labels_table{4, dataset_idx} = Y_predict_adaboost;
                    save(['target/' dataset_name '_' classifier], 'Y_predict_adaboost', 'Y_test_adaboost', 'n_steps_opt');

                case 'ann'
                    % Artificial Neural Network Optimal Parameter Testing Procedure
                    fprintf('\nTesting optimal artificial neural network model...\n');
                    
                    [~, ann_optimal_parameter_idx] = min(ann_parameter_errors(:));
                    [~, parameter_setings_idx] = ind2sub(size(ann_parameter_errors),ann_optimal_parameter_idx);
                    
                    n_hidden_units_opt = hidden_units(parameter_setings_idx);
                    momentum_opt = momentums(parameter_setings_idx);

                    x_tr = X_train';
                    y_tr = [1 - Y_train' ; Y_train'];
                    x_te = X_test';

                    net = patternnet(n_hidden_units_opt);
                    net.trainFcn = 'traingdm';   % use gradient descent with momentum
                    net.trainParam.mc = momentum_opt;
                    net.trainParam.showWindow = 1;
                    net.divideParam.trainRatio = 1; % use my own train,cv, and test set
                    net.divideParam.valRatio = 0;
                    net.divideParam.testRatio = 0;

                    [net,~] = train(net,x_tr,y_tr);
                    y_prob = net(x_te);
                    y_pred = vec2ind(y_prob) - 1;

                    Y_test_nn = Y_test;
                    Y_predict_nn = y_pred';
                    classifier_dataset_prediction_labels_table{5, dataset_idx} = Y_predict_nn;
                    save(['target/' dataset_name '_' classifier], 'Y_predict_nn', 'Y_test_nn', 'momentum_opt', 'n_hidden_units_opt');
            end
        end
    end
    
     save('target/classifier_dataset_prediction_labels_table',...
          'classifier_dataset_prediction_labels_table');
end