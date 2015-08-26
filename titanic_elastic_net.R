#####################################################################
# "R for Everyone", Jared P. Lander, (c) 2014, pp. 217-295          #
# Chapter 19: Regularization and Shrinkage                          #
# 19.1 Elastic Net                                                  #
#####################################################################

setwd("/Volumes/HD2/Users/pstessel/Documents/Git_Repos/Titanic")

# clear global environment
rm(list=ls(all = TRUE))

require(caret)
require(useful)
require(glmnet)
require(parallel)
require(doParallel)
require(reshape2)
require(stringr)
require(ROCR)
require(rattle)
require(randomForest)
require(dplyr)
require(lubridate)
require(Fselector)
require(tidyr)
rattle()

acs <- read.table("data/titanic_train.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

dsname <- "acs"
ds <- get(dsname)
dim(ds)

# build a data.frame where the first three columns are numeric
testFrame <-
  data.frame(First=sample(1:10, 20, replace=TRUE),
             Second=sample(1:20, 20, replace=TRUE),
             Third=sample(1:10, 20, replace=TRUE),
             Fourth=factor(rep(c("Alice", "Bob", "Charlie", "David"),
                               5)),
             Fifth=ordered(rep(c("Edward", "Frank", "Gerogia", "Hank", "Isaac"), 4)),
             Sixth=rep(c("a", "b"), 10), stringsAsFactors = F)
head(testFrame)
head(model.matrix(First ~ Second + Fourth + Fifth, testFrame))

# Not creating an indicator variable for the base level of a factor is essential
# for most linear models to avoid multicollinearity. However, it is generally
# considered undesirable for the predictor matrix to be designed this way for
# the Elastic Net.

# always use all levels
head(build.x(First ~ Second + Fourth + Fifth, testFrame, contrasts=FALSE))

# just use all levels for Fourth
head(build.x(First ~ Second + Fourth + Fifth, testFrame, contrasts=c(Fourth=FALSE, Fifth=TRUE)))

# Step 1: Load--Dataset ----------------------------------------------------

dspath <- "/Volumes/HD2/Users/pstessel/Documents/Git_Repos/Titanic/data/titanic_train.csv"
titanic <- read.csv(dspath)
dim(titanic)
names(titanic)
str(titanic)

# Step 1: Load--Generic Variables ------------------------------------------

# We will store the dataset as the generic variable ds (short for dataset). This
# will make the following steps somewhat generic and often we can just load a
# different dataset into ds and these steps can simply be re-run without change.

dsname <- "titanic"
ds <- get(dsname)
dim(ds)

# Step 1: Convenience of Table Data Frame ---------------------------------

# Another tip in dealing with larger datasets is to make use of tbl df() to add
# a couple of extra classes to the data frame. The simple aim here is to avoid
# the often made "mistake" of printing the whole data frame accidentally.

class(ds)
ds <-tbl_df(ds)

# Step 2: Review—Observations ---------------------------------------------

# Once we have loaded the dataset, the next step is to understand the shape of
# the dataset. We review the data using head() and tail() to get our first feel
# for the observations contained in the dataset. We also have a look at some
# random observations from the dataset to provide further insight.

head(ds)
tail(ds)
ds[sample(nrow(ds),6),]


# Step 2: Review—Structure --------------------------------------------------------

# Next we use str() to report on the structure of the dataset. Once again we get
# an overview of what the data looks like, and also now, how it is stored.

str(ds)

# Review—Summary ----------------------------------------------------------

# We use summary() to preview the distributions

summary(ds)

# Step 2: Review — Meta Data Cleansing --------------------------------------

# We demonstrate some meta-data changes here.

# Normalise Variable Names
# Sometimes it is convenient to map all variable names to low- ercase. R is case
# sensitive, so doing this does change the variable names. This can be useful
# when different upper/lower case conventions are intermixed in names like
# Incm_tax_PyBl and remembering how to capitalise when interactively exploring
# the data with 1,000 such variables is an annoyance. We often see such variable
# names arising when we import data from databases which are often case
# insensitive. Here we use normVarNames() from rattle, which attempts to do a
# reasonable job of converting variables from a dataset into a standard form.

names(ds)
names(ds) <- normVarNames(names(ds))

# Step 2: Review  — Data Formats ------------------------------------------

# We may want to correct the format of some of the variables in our dataset. We might first
# check the data type of each variable.

# Step 2: Review — Variable Roles -----------------------------------------

# We are now in a position to identify the roles played by the variables within
# the dataset. From our observations so far we note that the first variable
# (Date) is not relevant, as is, to the modelling (we could turn it into a
# seasonal variable which might be useful). Also we remove the second variable
# (Location) as in the data here it is a constant. We also identify the risk
# variable, if it is provided|it is a measure of the amount of risk or the
# importance of an observation with respect to the target variable. The risk is
# an output variable, and thus should not be used as an input to the modelling.

(vars <- names(ds))

target <- "survived"
id <- c("passenger_id", "x_name")

# Step 3: Clean — Ignore IDs, Outputs, Missing ----------------------------

# We will want to ignore some variables that are irrelevant or inappropriate for modelling.

# IDs and Outputs
# We start with the identifiers and the risk variable (which is an output
# variable). These should play no role in the modelling. Always watch out for
# including output variables as inputs to the modelling. This is one trap I
# regularly see from beginners.

ignore <- union(id, if (exists("risk")) risk)

# We might also identify any variable that has a unique value for every
# observation. These are sometimes identifiers as well and if so are candidates
# for ignoring.

(ids <- which(sapply(ds, function(x) length(unique(x))) == nrow(ds)))

ignore <- union(ignore, names(ids))

# All Missing
# We then remove any variables where all of the values are missing. There are
# none like this in the weather dataset, but in general across 1,000 variables,
# there may be some. We first count the number of missing values for each
# variable, and then list the names of those variables with only missing values.

mvc <- sapply(ds[vars], function(x) sum(is.na(x)))
mvn <- names(which(mvc == nrow(ds)))
ignore <- union(ignore, mvn)

# Many Missing
# Perhaps we also want to ignore variables with more than 70% of the values missing.

mvn <- names(which(mvc >= 0.7*nrow(ds)))
ignore <- union(ignore, mvn)

# Step 3: Clean — Ignore MultiLevel, Constants ----------------------------

# Too Many Levels
# We might also want to ignore variables with too many levels. Another approach
# is to group the levels into a smaller number of levels, but here we simply
# ignore them

factors <- which(sapply(ds[vars], is.factor))
lvls <- sapply(factors, function(x) length(levels(ds[[x]])))
(many <- names(which(lvls > 20)))

# Constants
# Ignore variables with constant values.

(constants <- names(which(sapply(ds[vars], function(x) all(x == x[1L])))))
ignore <- union(ignore, constants)

# Step 3: Clean — Identify Corelated Variables ----------------------------

mc <- cor(ds[which(sapply(ds, is.numeric))], use="complete.obs")
mc[upper.tri(mc, diag=TRUE)] <- NA
mc <-
  mc %>%
  abs() %>%
  data.frame() %>%
  mutate(var1=row.names(mc)) %>%
  gather(var2, cor, -var1) %>%
  na.omit()
mc <- mc[order(-abs(mc$cor)),]
mc

# Step 3: Clean — Remove the Variables ------------------------------------

# Once we have identified the variables to ignore, we remove them from our list of
# variables to use.

length(vars)

vars <- setdiff(vars, ignore)
length(vars)

ds <- ds[vars]
ds$survived <- with(ds, survived==1)
ds$pclass <- as.factor(ds$pclass)
ds$survived
names(ds)
str(ds)

table(ds$age, useNA = "ifany")
hist(ds$age)
quantile(ds$age, probs = seq(0, 1, 0.1), na.rm = TRUE)
boxplot(ds$age)
summary(ds$age)
?quantile

ds = transform(ds, age = ifelse(is.na(age), mean(age, na.rm=TRUE), y))



# Here we can identify pairs where we want to keep one but not the other,
# because they are highly correlated. We will select them manually since it is a
# judgement call. Normally we might limit the removals to those correlations
# that are 0.95 or more.

# ignore <- union(ignore, c("temp_3pm1", "Pressure_9am", "temp_9am"))



# build predictor matrix
# do not include the intercept as glmnet will add that automatically
dsX <-
  build.x(
    survived ~ pclass + sex + age + sib_sp + parch + fare + embarked -
      1, data = ds, contrasts = FALSE
  )

# check class and dimensions
class(dsX)
dim(acsX)
topleft(acsX, c=6)
topright(acsX, c=6)

# build response predictor
acsY <-
  build.y(
    Income ~ NumBedrooms + NumChildren + NumPeople + NumRooms + NumUnits + NumVehicles + NumWorkers + OwnRent + YearBuilt + ElectricBill + FoodStamp + HeatingFuel + Insurance + Language -
      1, data = acs)

head(acsY)
tail(acsY)

set.seed(1863561)
# run the cross-validated glmnet
acsCV1 <- cv.glmnet(x = acsX, y = acsY, family = "binomial", nfold = 5)

# The most important information returned from cv.glmnet are the
# cross-validation error and which value of lambda minimizes the
# cross-validation error. Additionally, it also returns the largest value of
# lambda with a cross-validation error that is within one standard error of the
# minimum. Theory suggests that the simpler model, even though it is slightly
# less accurate, should be preferred due to its parsimony.

acsCV1$lambda.min
acsCV1$lambda.1se

plot(acsCV1)

# Extracting the coefficients is done as with any other model, by using coef,
# except that a specific level of lambda should be specified; otherwise, the
# entire path is returned. Dots represent the variables that were not selected.

coef(acsCV1, s = "lambda.1se")

# Notice there are no standard errors and hence no confidence intervals for the
# coefficients. This is due to the theoretical properties of the lasso and
# ridge, and is an open problem.

# Visualizing where variables enter the model along the lambda path can be illuminating.

# plot the path
plot(acsCV1$glmnet.fit, xvar = "lambda")
# add in vertical lines for the optimal values of lambda
abline(v = log(c(acsCV1$lambda.min, acsCV1$lambda.1se)), lty=2)

# Setting alpha to 0 causes the results to be from the ridge. In this case,
# every variable is kept in the model but is just shrunk closer to 0.

# fit the ridge model
set.seed(71623)
acsCV2 <- cv.glmnet(x = acsX, y = acsY, family = "binomial", nfold = 5, alpha = 0)

# look at the lambda values
acsCV2$lambda.min
acsCV2$lambda.1se

# look at the coefficients
coef(acsCV2, s = "lambda.1se")

# The following plots the cross-validation curve

# plot the cross-validation error path
plot(acsCV2)

# Notice on the following plot that for every value of lambda there are still
# all the variables, just at different sizes

# plot the coefficient path
plot(acsCV2$glmnet.fit, xvar = "lambda")
abline(v = log(c(acsCV2$lambda.min, acsCV2$lambda.1se)), lty = 2)

# Finding the optimal value of alpha requires an additional layer of
# cross-validation, which glmnet does not automatically do. This requires
# running cv.glmnet at various levels of alpha, which will take a farily large
# chunk of time if performed sequentially, making this a good time to use
# parallelization. The most straightforward way to run code in parallel is to
# use the parallel, doParallel and foreach packages.

# First, we build some helper objects to speed along the process. When a
# two-layered cross-validation is run, an observation should fall in the same
# fold each time, so we build a vector specifying fold membership. We also
# specify the sequence of alpha values that foreach will loop over. It is
# generally considered better to lean toward the lasso rather than the ridge, so
# we consider only alpha values greater than 0.5.

# set the seed for repeatability of random results
set.seed(2834673)

# create folds, we want observations to be in the same fold each time it is run
theFolds <- sample(rep(x=1:5, length.out = nrow(acsX)))

# make sequence of alpha values
alphas <- seq(from = 0.5, to = 1, by = 0.05)

# Before running a parallel job, a cluster (even on a single machine) must be
# started and registered with makeCluster and registerDoParallel. After the job
# is done the cluster should be stopped with stopCluster. Setting .errorhandling
# to ''remove'' means that if an error occurs, that iteration will be skipped.
# Setting .inorder to FALSE means that the order of combining the results does
# not matter and they can be combined whenever returned, which yields
# significant speed improvements. Because we are using the default combination
# function, list, which takes multiple arguments at once, we can speed up the
# process by setting .multicombine to TRUE. We specify in .packages that glmnet
# should be loaded on each of the workers, again leading to performance
# improvements. The operator %dopar% tells foreach to work in parallel. Parallel
# computing can be dependent on the environment, so we explicitly load some
# variables into the foreach environment using .export, names, acsX, acsY,
# alphas and theFolds.

# set the seed for repeatability of random results
set.seed(5127151)

# start a cluster with two workers
cl <- makeCluster(2)
# register the workers
registerDoParallel(cl)

# keep track of timing
before <- Sys.time()

# build foreach loop to run in parallel
## several arguments
acsDouble <- foreach(i=1:length(alphas), .errorhandling = "remove", .inorder = FALSE, .multicombine = TRUE, .export = c("acsX", "acsY", "alphas", "theFolds"), .packages = "glmnet") %dopar%
{
  print(alphas[i])
  cv.glmnet(x=acsX, y=acsY, family="binomial", nfolds=5, foldid=theFolds, alpha=alphas[i])
}

# stop timing
after <- Sys.time()

# make sure to stop the culster when done
stopCluster(cl)

# time difference
# this will depend on speed, memory & number of cores of the machine
after - before

# Results in acsDouble should be a list with ll instances of cv.glmnet objects. We use sapply to check the class of each element of the list.

sapply(acsDouble, class)

# The goal is to find the best combination of lambda and alpha, so we need to build some code to extract the cross-validation error (including the confidence interval) and lambda from each element of the list.

# function for extracting info from cv.glmnet object
extractGlmnetInfo <- function(object)
{
  # find lambda
  lambdaMin <- object$lambda.min
  lambda1se <- object$lambda.1se

  # figure out where those lambdas fall in the path
  whichMin <- which(object$lambda == lambdaMin)
  which1se <- which(object$lambda == lambda1se)

  # build a one line data.frame with each of the selected lambdas and its corresponding error figures
  data.frame(lambda.min=lambdaMin, error.min=object$cvm[whichMin],
             lambda.1se=lambda1se, error.1se=object$cvm[which1se])
}

# apply that function to each element of the list
# combine it all into a data.frame
alphaInfo <- Reduce(rbind, lapply(acsDouble, extractGlmnetInfo))
alphaInfo

# could also be dine with ldply from plyr
alphaInfo2 <- plyr::ldply(acsDouble, extractGlmnetInfo)
identical(alphaInfo, alphaInfo2)

# make a column listing the alphas
alphaInfo$Alpha <- alphas
alphaInfo

# Now we plot this to pick out the best combination of alpha and lambda, which
# is where the plot shows minimum error. The following plot indicates that by
# using the one standard error methodology, the optimal alpha and lambda are
# 0.75 and 0.0054284, respectively.

## prepare the data.frame for plotting multiple pieces of information

# melt the data into long format
alphaMelt <- melt(alphaInfo, id.vars = "Alpha", value.name="Value", variable.name="Measure")
?melt
alphaMelt$Type <- str_extract(string=alphaMelt$Measure, pattern="(min)|(1se)")
?str_extract
# some housekeeping
alphaMelt$Measure <- str_replace(string=alphaMelt$Measure, pattern="\\.(min|1se)", replacement="")

alphaCast <- dcast(alphaMelt, Alpha + Type ~ Measure, value.var="Value")

ggplot(alphaCast, aes(x=Alpha, y=error)) +
  geom_line(aes(group=Type)) +
  facet_wrap(~Type, scales="free_y", ncol=1) +
  geom_point(aes(size=lambda))

## Now that we have found the optimal value of alpha (0.75), we refit the model and check he results.

set.seed(5127151)
acsCV3 <- cv.glmnet(x = acsX, y = acsY, family = "binomial", nfold = 5, alpha = alphaInfo$Alpha[which.min(alphaInfo$error.1se)])

acsCV3 <- cv.glmnet(x = acsX, y = acsY, type.measure = "auc", family = "binomial", nfold = 5, alpha = alphaInfo$Alpha[which.min(alphaInfo$error.1se)])

plot(acsCV3)

plot(acsCV3$glmnet.fit, xvar = "lambda")
abline(v = log(c(acsCV3$lambda.min, acsCV3$lambda.1se)), lty = 2)

## Viewing the coefficient plot for aglmnet object is not yet implemented in coefplot, so we build it manually. The next figure shows that the number of workers in the family and not being on foodstamps are the strongest indicators of having high income, and using coal heat and living in a mobile home are the strongest indicators of having low income. There are no standard errors because glmnet does not calculate them.

theCoef <- as.matrix(coef(acsCV3, s = "lambda.1se"))
coefDF <- data.frame(Value = theCoef, Coefficient = rownames(theCoef))
coefDF <- coefDF[nonzeroCoef(coef(acsCV3, s = "lambda.1se")),]
ggplot(coefDF, aes(x = X1, y = reorder(Coefficient, X1))) +
  geom_vline(xintercept = 0, color = "grey", linetype = 2) +
  geom_point(color = "blue") + labs(x = "Value", y = "Coefficient", title = "Coefficient Plot")

### MODEL SCORING

pred <- predict(acsCV3, newx = acsX, family = "binomial", s = "lambda.min", type = "class")

confusionMatrix(pred, acs$Income)

### ROC CURVE


# Run model over training dataset
acsCV3_AUC <- cv.glmnet(x = acsX, y = acsY, type.measure = "auc", family = "binomial", nfold = 5, alpha = alphaInfo$Alpha[which.min(alphaInfo$error.1se)])

# Apply model to testing dataset
acsCV3_AUC.prob <- predict(acsCV3_AUC,type="response",
                           newx = acsX, s = 'lambda.min')
pred_AUC <- prediction(acsCV3_AUC.prob, acs$Income)

# calculate probabilities for TPR/FPR for predictions
perf <- performance(pred_AUC,"tpr","fpr")
performance(pred_AUC,"auc") # shows calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

