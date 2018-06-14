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


train_df = titanic_train %>% select(-Name)
test_df = titanic_test %>% select(-Name)

X_train = train_df %>% select(-Survived)
y_train = train_df %>% select(Survived)
X_test = test_df %>% select(-Survived)
y_test = test_df %>% select(Survived)




