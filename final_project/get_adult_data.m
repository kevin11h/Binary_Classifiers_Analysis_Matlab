% http://archive.ics.uci.edu/ml/datasets/Adult
function [X, Y, K_classes] = get_adult_data()
    if exist('datasets/adult.mat')
        load('datasets/adult');
        X = adultX;
        Y = adultY;
        K_classes = 2;
        return;
    end

    adult_tr_data_fid = fopen('datasets/adult_tr.data');
    t =  textscan(adult_tr_data_fid, '%d%s%d%s%d%s%s%s%s%s%d%d%d%s%s', 'delimiter', ',');
    age_tr = t{1};
    workclass_tr = t{2};  
    fnlwgt_tr = t{3};
    education_tr = t{4};  
    education_num_tr = t{5}; 
    marital_status_tr = t{6};  
    occupation_tr = t{7};  
    relationship_tr = t{8};  
    race_tr = t{9};  
    sex_tr = t{10};  
    capital_gain_tr = t{11};
    capital_loss_tr = t{12};
    hours_per_week_tr = t{13};
    native_country_tr = t{14};  
    census_income_tr = t{15};
    fclose(adult_tr_data_fid);
    
    adult_te_data_fid = fopen('datasets/adult_te.data');
    t =  textscan(adult_te_data_fid, '%d%s%d%s%d%s%s%s%s%s%d%d%d%s%s', 'delimiter', ',');
    age_te = t{1};
    workclass_te = t{2};  
    fnlwgt_te = t{3};
    education_te = t{4};  
    education_num_te = t{5}; 
    marital_status_te = t{6};  
    occupation_te = t{7};  
    relationship_te = t{8};  
    race_te = t{9};  
    sex_te = t{10};  
    capital_gain_te = t{11};
    capital_loss_te = t{12};
    hours_per_week_te = t{13};
    native_country_te = t{14};  
    census_income_te = t{15};
    fclose(adult_te_data_fid);
    
    age = [age_tr ; age_te];
    workclass = [workclass_tr ; workclass_te];
    fnlwgt = [fnlwgt_tr ; fnlwgt_te];
    education = [education_tr ; education_te]; 
    education_num = [education_num_tr ; education_num_te];
    marital_status = [marital_status_tr ; marital_status_te];
    occupation = [occupation_tr ; occupation_te];
    relationship = [relationship_tr ; relationship_te];
    race = [race_tr ; race_te];
    sex = [sex_tr ; sex_te];
    capital_gain = [capital_gain_tr ; capital_gain_te];
    capital_loss = [capital_loss_tr ; capital_loss_te];
    hours_per_week = [hours_per_week_tr ; hours_per_week_te];
    native_country = [native_country_tr ; native_country_te];
    census_income = [census_income_tr ; census_income_te];
    
    m = size(census_income, 1);
    
    X1 = age < median(single(age));
    
    workclass_values = {'Private', 'Self-emp-not-inc', 'Self-emp-inc',...
                        'Federal-gov', 'Local-gov', 'State-gov',...
                        'Without-pay', 'Never-worked'};
    
    X2 = zeros(m, length(workclass_values));
    
    for i = 1:m
       w = workclass(i);
       j = find(cellfun(@(v) strcmp(v, w) == 1, workclass_values));
       X2(i, j) = 1;
    end
    
    X3 = fnlwgt < median(double(fnlwgt));
    
    education_values = {'Bachelors', 'Some-college', '11th', 'HS-grad',...
                        'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',...
                        '7th-8th', '12th', 'Masters', '1st-4th', '10th',...
                        'Doctorate', '5th-6th', 'Preschool'};
    
    X4 = zeros(m, length(education_values));
                    
    for i = 1:m
       e = education(i);
       j = find(cellfun(@(v) strcmp(v, e) == 1, education_values));
       X4(i, j) = 1;
    end
    
    X5 = education_num < median(single(education_num));
    
    marital_status_values = {'Married-civ-spouse', 'Divorced',...
                             'Never-married', 'Separated', 'Widowed',...
                             'Married-spouse-absent', 'Married-AF-spouse'};
                         
    X6 = zeros(m, length(marital_status_values));
    
    for i = 1:m
       ms = marital_status(i);
       j = find(cellfun(@(v) strcmp(v, ms) == 1, marital_status_values));
       X6(i, j) = 1;
    end
    
    occupation_values = {'Tech-support', 'Craft-repair','Other-service',...
                         'Sales', 'Exec-managerial', 'Prof-specialty',...
                         'Handlers-cleaners', 'Machine-op-inspct',...
                         'Adm-clerical', 'Farming-fishing',...
                         'Transport-moving', 'Priv-house-serv',...
                         'Protective-serv', 'Armed-Forces'};
                     
    X7 = zeros(m, length(occupation_values));
    
    for i = m
        ov = occupation(i);
        j = find(cellfun(@(v) strcmp(v, ov) == 1, occupation_values));
        X7(i, j) = 1;
    end
    
    relationship_values = {'Wife', 'Own-child', 'Husband',...
                           'Not-in-family', 'Other-relative', 'Unmarried'};
                       
    X8 = zeros(m, length(relationship_values));
    
    for i = 1:m
        rv = relationship(i);
        j = find(cellfun(@(v) strcmp(v, rv) == 1, relationship_values));
        X8(i, j) = 1;
    end
    
    race_values = {'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',...
                   'Other', 'Black'};
               
    X9 = zeros(m, length(race_values));
    
    for i = 1:m
        r = race(i);
        j = find(cellfun(@(v) strcmp(v, r) == 1, race_values));
        X9(i, j) = 1;
    end
    
    sex_values = {'Female', 'Male'};
    
    X10 = strcmp(sex, 'Female');
    X11 = capital_gain < median(single(capital_gain));
    X12 = capital_loss < median(single(capital_loss));
    X13 = hours_per_week < median(single(hours_per_week));
    
    native_country_values = {'United-States', 'Cambodia', 'England',...
                             'Puerto-Rico', 'Canada', 'Germany',...
                             'Outlying-US(Guam-USVI-etc)', 'India',...
                             'Japan', 'Greece', 'South', 'China', 'Cuba',...
                             'Iran', 'Honduras', 'Philippines', 'Italy',...
                             'Poland', 'Jamaica', 'Vietnam', 'Mexico',...
                             'Portugal', 'Ireland', 'France',...
                             'Dominican-Republic', 'Laos', 'Ecuador',...
                             'Taiwan', 'Haiti', 'Columbia', 'Hungary',...
                             'Guatemala', 'Nicaragua', 'Scotland',...
                             'Thailand', 'Yugoslavia', 'El-Salvador',...
                             'Trinadad&Tobago', 'Peru', 'Hong',...
                             'Holand-Netherlands'};
                         
    X14 = zeros(m, length(native_country_values));
    
    for i = 1:m
        nc = native_country(i);
        j = find(cellfun(@(v) strcmp(v, nc) == 1, native_country_values));
        X14(i, j) = 1;
    end
    
    X = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14];
       
    X(~isfinite(X)) = 0;
    
    Y = zeros(size(census_income));
    Y(find(strcmp(census_income, '<=50K'))) = 0;
    Y(find(strcmp(census_income, '>50K'))) = 1;
    
    K_classes = length(unique(Y));
    
    adultX = X;
    adultY = Y;
    save('datasets/adult', 'adultX', 'adultY');
end
