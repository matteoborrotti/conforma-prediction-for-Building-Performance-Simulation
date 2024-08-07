# Clean workspace
rm(list = c())

library("conformalInference")
library(ggplot2)
library(randomForest)
library(caret)
library(e1071)
library(neuralnet)

source("svm.funs.R")
source("neuralnet.funs.R")

# Read external file
energy <- read.csv2("ENB2012_data.csv")
# Set a seed for reproducibility
set.seed(123)

# Split the dataset into training (80%) and test (20%) sets
trainIndex <- createDataPartition(energy$Y2,
                                  p = 0.8,
                                  list = FALSE,
                                  times = 1)

energy_train <- energy[trainIndex, ]
energy_test  <- energy[-trainIndex, ]

p <- ncol(energy_train) - 2
n <- nrow(energy_train)
n_new <- nrow(energy_test)

x_train <- energy_train[, 1:p]
average <- colMeans(x_train)
standard <- apply(x_train, 2, sd)
x_train <- scale(x_train)
y_train <- energy_train$Y1

x_test <- energy_test[, 1:p]
x_test <- scale(x_test, center = average, scale = standard)
y_test <- energy_test$Y1



cov.confpred <- NULL
len.confpred <- NULL
err.confpred <- NULL
train.confpred <- NULL

B <- 50
alpha_value <- 0.1

out.split.rf <- NULL
out.split.nn <- NULL
out.split.svm <- NULL
out.split.step <- NULL

cov.split.rf <- NULL
cov.split.nn <- NULL
cov.split.svm <- NULL
cov.split.step <- NULL

my.rf.funs = rf.funs(ntree = 500, varfrac = 0.333)
my.neuralnet.funs = neuralnet.funs(size = p)
my.svm.funs = svm.funs(kernel_type = "linear")
my.step.funs = lars.funs(type = "stepwise", max.steps = 20)

for (b in 1:B) {
  print(b)
  set.seed(b)
  
  output.rf <-
    conformal.pred.split(
      x_train,
      y_train,
      as.matrix(x_test),
      alpha = alpha_value,
      rho = 0.5,
      verb = "\t\t",
      train.fun = my.rf.funs$train,
      predict.fun = my.rf.funs$predict
    )
  
  output.nn <-
    conformal.pred.split(
      x_train,
      y_train,
      as.matrix(x_test),
      alpha = alpha_value,
      rho = 0.5,
      verb = "\t\t",
      train.fun = my.neuralnet.funs$train,
      predict.fun = my.neuralnet.funs$predict
    )
  
  output.svm <-
    conformal.pred.split(
      x_train,
      y_train,
      as.matrix(x_test),
      alpha = alpha_value,
      rho = 0.5,
      verb = "\t\t",
      train.fun = my.svm.funs$train,
      predict.fun = my.svm.funs$predict
    )
  
  output.step <-
    conformal.pred.split(
      x_train,
      y_train,
      as.matrix(x_test),
      alpha = alpha_value,
      rho = 0.5,
      verb = "\t\t",
      train.fun = my.step.funs$train,
      predict.fun = my.step.funs$predict
    )
  
  out.split.rf[["pred"]] <-
    cbind(out.split.rf[["pred"]], output.rf$pred)
  out.split.rf[["lo"]] <-
    cbind(out.split.rf[["lo"]], output.rf$lo)
  out.split.rf[["up"]] <-
    cbind(out.split.rf[["up"]], output.rf$up)
  
  out.split.nn[["pred"]] <-
    cbind(out.split.nn[["pred"]], output.nn$pred)
  out.split.nn[["lo"]] <-
    cbind(out.split.nn[["lo"]], output.nn$lo)
  out.split.nn[["up"]] <-
    cbind(out.split.nn[["up"]], output.nn$up)
  
  out.split.svm[["pred"]] <-
    cbind(out.split.svm[["pred"]], output.svm$pred)
  out.split.svm[["lo"]] <-
    cbind(out.split.svm[["lo"]], output.svm$lo)
  out.split.svm[["up"]] <-
    cbind(out.split.svm[["up"]], output.svm$up)
  
  out.split.step[["pred"]] <-
    cbind(out.split.step[["pred"]], output.step$pred[, p])
  out.split.step[["lo"]] <-
    cbind(out.split.step[["lo"]], output.step$lo[, p])
  out.split.step[["up"]] <-
    cbind(out.split.step[["up"]], output.step$up[, p])
  
}
y0.mat = matrix(rep(y_test, B), nrow = length(y_test))

cov.split.rf = colMeans(out.split.rf$lo <= y0.mat &
                          y0.mat <= out.split.rf$up)
len.split.rf = colMeans(out.split.rf$up - out.split.rf$lo)
err.split.rf = colMeans((y0.mat - as.matrix(out.split.rf$pred)) ^ 2)

cov.split.nn = colMeans(out.split.nn$lo <= y0.mat &
                          y0.mat <= out.split.nn$up)
len.split.nn = colMeans(out.split.nn$up - out.split.nn$lo)
err.split.nn = colMeans((y0.mat - as.matrix(out.split.nn$pred)) ^ 2)

cov.split.svm = colMeans(out.split.svm$lo <= y0.mat &
                           y0.mat <= out.split.svm$up)
len.split.svm = colMeans(out.split.svm$up - out.split.svm$lo)
err.split.svm = colMeans((y0.mat - as.matrix(out.split.svm$pred)) ^ 2)

cov.split.step = colMeans(out.split.step$lo <= y0.mat &
                            y0.mat <= out.split.step$up)
len.split.step = colMeans(out.split.step$up - out.split.step$lo)
err.split.step = colMeans((y0.mat - as.matrix(out.split.step$pred)) ^ 2)

average.mse <- c(mean(err.split.step), mean(err.split.svm), mean(err.split.rf), mean(err.split.nn))
sd.mse <- c(sd(err.split.step), sd(err.split.svm), sd(err.split.rf), sd(err.split.nn))

average.cov <- c(mean(cov.split.step), mean(cov.split.svm), mean(cov.split.rf), mean(cov.split.nn))
sd.cov <- c(sd(cov.split.step), sd(cov.split.svm), sd(cov.split.rf), sd(cov.split.nn))

average.len <- c(mean(len.split.step), mean(len.split.svm), mean(len.split.rf), mean(len.split.nn))
sd.len <- c(sd(len.split.step), sd(len.split.svm), sd(len.split.rf), sd(len.split.nn))

write.table(rbind(average.mse, sd.mse), "mse_y1.txt")
write.table(rbind(average.cov, sd.cov), "cov_y1.txt")
write.table(rbind(average.len, sd.len), "len_y1.txt")

mse.step <- NULL
mse.svm <- NULL
mse.nn <- NULL
mse.rf <- NULL

mse.step <-
  as.data.frame(cbind("MSE" = err.split.step, "Models" = rep("Stepwise", B)))

mse.svm <-
  as.data.frame(cbind("MSE" = err.split.svm, "Models" = rep("SVM", B)))

mse.nn <-
  as.data.frame(cbind("MSE" = err.split.nn, "Models" = rep("Neural Net", B)))

mse.rf <-
  as.data.frame(cbind("MSE" = err.split.rf, "Models" = rep("Random Forest", B)))


mse <- rbind(mse.step, mse.svm, mse.nn, mse.rf)

mse$MSE <- as.numeric(mse$MSE)
p.mse <- ggplot(mse, aes(x=Models, y=MSE)) + 
  labs(x = "Models", y = "Test MSE") +
  geom_boxplot()
png("mse_y1.png")
print(p.mse)
dev.off()



cov.step <- NULL
cov.svm <- NULL
cov.nn <- NULL
cov.rf <- NULL

cov.step <-
  as.data.frame(cbind("Coverage" = cov.split.step, "Models" = rep("Stepwise", B)))

cov.svm <-
  as.data.frame(cbind("Coverage" = cov.split.svm, "Models" = rep("SVM", B)))

cov.nn <-
  as.data.frame(cbind("Coverage" = cov.split.nn, "Models" = rep("Neural Net", B)))

cov.rf <-
  as.data.frame(cbind("Coverage" = cov.split.rf, "Models" = rep("Random Forest", B)))


cov <- rbind(cov.step, cov.svm, cov.nn, cov.rf)

cov$Coverage <- as.numeric(cov$Coverage)
p.cov <- ggplot(cov, aes(x=Models, y=Coverage)) + 
  labs(x = "Models", y = "Empirical coverage") +
  geom_boxplot()
png("cov_y1.png")
print(p.cov)
dev.off()

