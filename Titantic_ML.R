library(dplyr)        ## Data Manipulation
library(modelr)       ## Partition Data set
library(class)        ## KNN
library(rpart)        ## CART
library(randomForest) ## Random Forest
library(xgboost)      ## XGBoost
library(e1071)        ## SVM
library(tidyr)        ## Organize Data
library(ggplot2)      ## View Data
library(titanic)      ## Get Data
library(stringr)



metrics_function = function(matrixM){
  ##---------------------------------------
  ## Condition positive (P) - The number of real positive cases in the data
  ##---------------------------------------
  P_cond = sum(matrixM[1,])
  ##---------------------------------------
  ## Condition negative (N) - The number of real negative cases in the data
  ##---------------------------------------
  N_cond = sum(matrixM[2,])
  ##---------------------------------------
  ## True positive (TP) - Eqv. with hit
  ##---------------------------------------
  TP_cond = matrixM[1,1]
  ##---------------------------------------
  ## True negative (TN) - Eqv. with correct rejection
  ##---------------------------------------
  TN_cond = matrixM[2,2]
  ##---------------------------------------
  ## False positive (FP) - Eqv. with false alarm, Type I error
  ##---------------------------------------
  FP_cond = matrixM[1,2]
  ##---------------------------------------
  ## False negative (FN) - Eqv. with miss, Type II error
  ##---------------------------------------
  FN_cond = matrixM[2,1]
  
  ## Metrics
  ## 1) Sensitivity/Recall/True Positive Rate
  TPR_metric = TP_cond / P_cond
  ## 2) SpecificityTrue Negative Rate
  TNR_metric = TN_cond / N_cond
  ## 3) Precision/ Positive Predictive Value
  PPV_metric = TP_cond / (TP_cond + FP_cond)
  ## 4) Negative Predictive value
  NPV_metric = TN_cond / (TN_cond + FN_cond)
  ## 5) False Negative Rate
  FNR_metric = FN_cond / P_cond
  ## 6) False Positive Rate
  FPR_metric = FP_cond / N_cond
  ## 7) False Discovery Rate
  FDR_metric = FP_cond / ( FP_cond + TP_cond)
  ## 8) False Omission Rate
  FOR_metric = FN_cond / (FN_cond + TN_cond)
  ## 9) Accuracy
  ACC_metric = (TP_cond + TN_cond)/ (P_cond + N_cond)
  ## 10) F1 Score
  F1s_metric = 2 *((PPV_metric*TPR_metric) / (PPV_metric + TPR_metric))
  ## 11) F2 Score
  F2s_metric =  5 *((PPV_metric*TPR_metric) / (4*PPV_metric + TPR_metric))
  
  ## 12) Matthews Correlation Cofficient
  #MCC_metric = ((TP_cond * TN_cond) - (FP_cond * FN_cond)) / 
  #  sqrt((TP_cond+FP_cond)*(TP_cond+FN_cond)*(TN_cond+FP_cond)*(TN_cond+FN_cond))
  
  ## 13) Informedness 
  BMi_metric = TPR_metric + TNR_metric - 1
  
  ## 14) MarKedness
  MKd_metric = PPV_metric + NPV_metric - 1
  
  
  metrics = cbind(TPR_metric, TNR_metric, PPV_metric, NPV_metric,
                  FNR_metric, FPR_metric, FDR_metric, FOR_metric, 
                  ACC_metric, F1s_metric, F2s_metric, #MCC_metric,
                  BMi_metric, MKd_metric)
  
  colnames(metrics) = c("TPR_metric", "TNR_metric", 
                        "PPV_metric", "NPV_metric",
                        "FNR_metric", "FPR_metric", 
                        "FDR_metric", "FOR_metric", 
                        "ACC_metric", "F1s_metric", 
                        "F2s_metric", #"MCC_metric",
                        "BMi_metric", "MKd_metric")
  return(metrics)
}




log_function = function(train_df,x_test,y_test){
  log_model = glm(y ~.,
                  data = train_df,
                  family="binomial")
  ## Predict for logistic regression
  est_log = ifelse(predict(log_model, data.frame(x_test), type = "response")>0.5,1,0)
  ## Confusion matrix for logistic regression
  m_log = table(y_test,est_log)
  m_log = data.frame(method = "Logistic_Regression",
                     metrics_function(m_log))
  return(m_log)
}

knn_function = function(x_train,x_test,y_train,y_test,knn_param){
  est_knn = knn(x_train,x_test,cl = y_train,k = knn_param)
  ## Confusion matrix for KNN
  m_knn = table(y_test,est_knn)
  m_knn = data.frame(method = "K-Nearest_Neighbors",
                     metrics_function(m_knn))
  return(m_knn)
}

cart_function = function(train_df, x_test, y_test, cart_split, cart_bucket){
  ## Run Model
  cart_model = rpart(y ~. , method="class", data = train_df, 
                     minsplit = cart_split, minbucket = cart_bucket)
  ## Predict for CART
  result_cart_test = predict(cart_model,data.frame(x_test))
  est_cart = apply(result_cart_test,1,which.max) - 1
  
  ## Confusion matrix for CART
  m_cart = table(y_test,est_cart)
  m_cart = data.frame(method = "CART",
                      metrics_function(m_cart))
  return(m_cart)
}

rf_function = function(train_df,x_test,y_test,rf_tree_number){
  rf_model = randomForest(as.factor(y) ~.,
                          data=train_df, 
                          importance=TRUE, 
                          ntree=rf_tree_number)
  ## Predict for logistic regression
  est_rf = predict(rf_model,data.frame(x_test))
  ## Confusion matrix for Random Forest
  m_rf = table(y_test,est_rf)
  m_rf = data.frame(method = "Random_Forest",
                    metrics_function(m_rf))
  return(m_rf)
}

svm_function = function(train_df,x_test,y_test,svm_kernel,svm_cost){
  svm_model = svm(y ~. , data = train_df, cost = svm_cost,
                  kernel = svm_kernel)
  ## Predict for SVM
  est_svm = ifelse(predict(svm_model,data.frame(x_test))>0.5,1,0)
  ## Confusion matrix for SVM
  m_svm = table(y_test,est_svm)
  m_svm = data.frame(method = "SVM",
                     metrics_function(m_svm))
  return(m_svm)
}

xgb_function = function(x_train,x_test,y_train,y_test,n_rounds){
  xgb_model <- xgboost(data = x_train, label = y_train, nrounds = n_rounds)
  ## Predict for xgboost
  est_xgb = ifelse(predict(xgb_model,x_test)>0.5,1,0)
  ## Confusion matrix for xgboost
  m_xgb = table(y_test,est_xgb)
  m_xgb = data.frame(method = "XGBoost",
                     metrics_function(m_xgb))
  return(m_xgb)
}

binary_ml = function(x,y,p,knn_param,svm_cost,svm_kernel,
                     cart_split = 2,
                     cart_bucket = 1,
                     rf_tree_number = 200, 
                     n_rounds = 10){ 
  
  
  df = data.frame(x,y)
  #######################
  ## 1) Preprocessing Data
  #######################
  ##--------
  # a) Partition Data set (Train & Test)
  ##--------
  p_test = p
  partitions = resample_partition(df,c(part0 = 1 - p_test,
                                       part1 = p_test))
  train_df = data.frame(partitions$part0)
  test_df  = data.frame(partitions$part1)
  
  
  ##--------
  # b) Re-format Data for different Shallow Algos
  ##--------
  x_train = as.matrix(train_df %>% select(-y))
  y_train = train_df %>% select(y)
  y_train = as.numeric(factor(y_train$y))-1
  
  x_test = as.matrix(test_df %>% select(-y))
  y_test = test_df %>% select(y)
  y_test = as.numeric(factor(y_test$y))-1
  
  #################################
  ## ML1: Logistic Regression
  #################################
  ## Run Model
  #m_log = log_function(train_df,x_test,y_test);
  
  #################################
  ## ML2: KNN
  #################################
  ## Run Model
  #m_knn = knn_function(x_train,x_test,y_train,y_test,knn_param)
  
  #################################
  ## ML3: CART
  #################################
  ## Run Model
  m_cart = cart_function(train_df,x_test,y_test,cart_split,cart_bucket)
  
  #################################
  ## ML4: Random Forest
  #################################
  ## Run Model
  m_rf = rf_function(train_df,x_test,y_test,rf_tree_number)
  
  #################################
  ## ML5: SVM
  #################################
  ## Run Model
  #m_svm = svm_function(train_df,x_test,y_test,svm_kernel,svm_cost)
  
  
  #################################
  ## ML6: XGBoost
  #################################
  ## Run Model
  m_xgb = xgb_function(x_train,x_test,y_train,y_test,n_rounds)
  
  m_all = rbind(m_log,m_cart,m_knn,m_rf,m_svm,m_xgb)
  return(m_all)
}

## Gender
titanic_train$Gender = ifelse(titanic_train$Sex == "male",1,0)

## Cabin Info
cabina_asg = as.matrix(gsub('[0-9]+', '', titanic_train$Cabin))
cabinfun = function(x){
  if(x %in% "A"){
    return(1)
  } else if(x %in% "B"){
    return(2)
  } else if(x %in% "C"){
    return(3)
  } else if(x %in% "D"){
    return(4)
  } else if(x %in% "E"){
    return(5)
  } else if(x %in% "F"){
    return(6)
  } else if(x %in% "G"){
    return(7)
  } else{
    return(0)
  }  
}
titanic_train$cabin_info = unlist(apply(cabina_asg,1,cabinfun))

## Embarked Info
new_titanic_train$embark = ifelse(titanic_train$Embarked == "C",0,
       ifelse(titanic_train$Embarked == "Q",1,2))


## Remove Non-Numeric Vales
new_titanic_train = titanic_train %>% 
  select(-Ticket,-Name,-Sex,-Cabin,-PassengerId,-Embarked)


x = new_titanic_train %>% select(-Survived)
y = new_titanic_train %>% select(Survived)
colnames(y) = "y"
met = binary_ml(x,y,p = 0.25,
                knn_param = 3,svm_cost = 1,
                svm_kernel = "radial")



