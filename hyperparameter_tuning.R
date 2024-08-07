# Clean workspace
rm(list = c())

library("conformalInference")
library(ggplot2)

library(randomForest)
library(caret)
library(e1071)


# Read external file
energy <- read.csv2("ENB2012_data.csv")


# Set a seed for reproducibility
set.seed(123)

# Split the dataset into training (80%) and test (20%) sets
trainIndex <- createDataPartition(energy$Y2,
                                  p = 0.8,
                                  list = FALSE,
                                  times = 1)

energy_train <- energy[ trainIndex,]
energy_test  <- energy[-trainIndex,]

n <- nrow(energy_train)
n_new <- nrow(energy_test)

x_train <- energy_train[, 1:8]
average <- colMeans(x_train)
standard <- apply(x_train, 2, sd)
x_train <- scale(x_train)
y_train <- energy_train$Y1

x_test <- energy_test[, 1:8]
x_test <- scale(x_test, center = average, scale = standard)
y_test <- energy_test$Y1



cov.rf.confpred <- NULL
len.rf.confpred <- NULL
err.rf.confpred <- NULL
train.rf.confpred <- NULL



B <- 50
alpha_value <- 0.1


features <- c(0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1)




for (f in features) {
  out.split<-NULL
  for (b in 1:B) {
    print(b)
    set.seed(b)
    my.rf.funs = rf.funs(ntree = 500, varfrac = f)
   
    
    output <-
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
    
     out.split[["pred"]] <- cbind(out.split[["pred"]], output$pred)
    #out.split[["pred_train"]] <- cbind(out.split[["pred_train"]], y_train)
    out.split[["lo"]] <- cbind(out.split[["lo"]], output$lo)
    out.split[["up"]] <- cbind(out.split[["up"]], output$up)
    
  }
  y0.mat = matrix(rep(y_test, B), nrow = length(output$lo))

  cov.split = colMeans(out.split$lo <= y0.mat &
                         y0.mat <= out.split$up)
  len.split = colMeans(out.split$up - out.split$lo)
  err.split = colMeans((y0.mat - as.matrix(out.split$pred)) ^ 2)

  cov.rf.confpred <- rbind(cov.rf.confpred, cov.split)
  len.rf.confpred <- rbind(len.rf.confpred, len.split)
  err.rf.confpred <- rbind(err.rf.confpred, err.split)

  
  
  
}

n<-length(y_test)

mean.cov <- rowMeans(cov.rf.confpred)
sd.cov <- apply(cov.rf.confpred, 1, sd)
se.cov <- sd.cov / sqrt(n)
ci_lower = mean.cov - qt(1 - (0.05 / 2), n - 1) * se.cov
ci_upper = mean.cov + qt(1 - (0.05 / 2), n - 1) * se.cov

coded_features <- c(2,3,4,5,6,7,8)
data <- data.frame(
  group = coded_features,
  mean = mean.cov,
  lower = ci_lower,
  upper = ci_upper
)

p.cov <- ggplot(data, aes(x = group, y = mean)) +
  geom_point(size = 3) +
  ylim(0.90, 0.93) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  labs(x = "m",
       y = "Empirical coverage") 
#+ theme_minimal()
png("y1_cov.png")
print(p.cov)
dev.off()


mean.len <- rowMeans(len.rf.confpred)
sd.len <- apply(len.rf.confpred, 1, sd)
se.len <- sd.len / sqrt(n)
ci_lower.len = mean.len - qt(1 - (0.05 / 2), n - 1) * se.len
ci_upper.len = mean.len + qt(1 - (0.05 / 2), n - 1) * se.len

data_len <- data.frame(
  group = coded_features,
  mean = mean.len,
  lower = ci_lower.len,
  upper = ci_upper.len
)

p.int <- ggplot(data_len, aes(x = group, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  labs(x = "m",
       y = "Pred. interval length") 
#+ theme_minimal()
png("y1_length.png")
print(p.int)
dev.off()

mean.err <- rowMeans(err.rf.confpred)
sd.err <- apply(err.rf.confpred, 1, sd)
se.err <- sd.err / sqrt(n)
ci_lower.err = mean.err - qt(1 - (0.05 / 2), n - 1) * se.err
ci_upper.err = mean.err + qt(1 - (0.05 / 2), n - 1) * se.err

data_err <- data.frame(
  group = coded_features,
  mean = mean.err,
  lower = ci_lower.err,
  upper = ci_upper.err
)

p.err<-ggplot(data_err, aes(x = group, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  labs(x = "m",
       y = "Test MSE") 
#+ theme_minimal()
png("y1_err.png")
print(p.err)
dev.off()