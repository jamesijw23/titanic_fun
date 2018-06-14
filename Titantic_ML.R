library(dplyr)
library(titanic)

train_df = titanic_train %>% select(-Name)
test_df = titanic_test %>% select(-Name)

X_train = train_df %>% select(-Survived)
y_train = train_df %>% select(Survived)
X_test = test_df %>% select(-Survived)
y_test = test_df %>% select(Survived)