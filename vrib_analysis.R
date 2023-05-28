library(ordinal)
library(tidyr)
library(dplyr)
library(lme4)
library(bayestestR)
library(emmeans)
library(ggplot2)

### Data ---------------------

data <- read.csv("avg_gaze_per_pas_long_intact.csv")
data$Subject <- factor(data$Subject)
data$subjectiveAwareness <- factor(data$subjectiveAwareness)
data$condition <- factor(data$condition)
data$objectiveIsCorrect <- factor(data$objectiveIsCorrect)


# Prepare results saving
sink(file="vrib_model.txt", append=TRUE)  # sink will output prints into a txt file instead of a console



## =================== ANALYSIS 1 =================== 
"
In this analysis, we test whether PAS (subjectiveAwareness) is different across conditions (condition), 
as well as test whether there is a relationship between PAS (subjectiveAwareness) and SOA (subjectiveTimeFromLastIntactSec). 
"

print("=================== ANALYSIS 1 =================== ")

## Models ------------------------

# Hypothesis
model1_h1 <- clmm(subjectiveAwareness ~ condition + subjectiveTimeFromLastIntactSec + (condition + subjectiveTimeFromLastIntactSec | Subject),
              data = data)
print("-----------------------------model 1 summary-----------------------------")
print(summary(model1_h1))

# Null i
model1_h0i <- clmm(subjectiveAwareness ~  subjectiveTimeFromLastIntactSec + (condition + subjectiveTimeFromLastIntactSec | Subject),
              data = data)
print("-----------------------------model 1h0i summary-----------------------------")
print(summary(model1_h0i))

# Null ii 
model1_h0ii <- clmm(subjectiveAwareness ~  condition + (condition + subjectiveTimeFromLastIntactSec | Subject),
                 data = data)
print("-----------------------------model 1h0ii summary-----------------------------")
print(summary(model1_h0ii))

## Comparisons ------------------------
"
We will compare the hypothesis model with each of the null models by calculating Bayes Factors based on the BIC approximation. 
"

print("-----------------------------model 1 comparisons: h1 v h0i -----------------------------")
# How much the alternative hypothesis is preferred over the null(i)?
print(bayesfactor_models(model1_h1, denominator = model1_h0i))  # this calculates the BIC for both models, then turns them into BFs


print("-----------------------------model 1 comparisons: h1 v h0ii -----------------------------")
# How much the alternative hypothesis is preferred over the null(ii)?
print(bayesfactor_models(model1_h1, denominator = model1_h0ii))


## Visualisation ------------------------

### Vis 1
# https://cran.r-project.org/web/packages/emmeans/vignettes/sophisticated.html#ordinal
ems <- emmeans(model1_h1, ~ condition + subjectiveTimeFromLastIntactSec + subjectiveAwareness ,
               at = list(subjectiveTimeFromLastIntactSec = seq(12, 70, len = 20)),
               mode = "prob")


ggplot(as.data.frame(ems), aes(subjectiveTimeFromLastIntactSec, prob, fill = subjectiveAwareness)) + 
  facet_grid(~ condition) + 
  geom_col() 


### Vis 2
ems <- emmeans(model1_h1, ~ condition + subjectiveTimeFromLastIntactSec ,
               at = list(subjectiveTimeFromLastIntactSec = seq(12, 70, len = 20)),
               mode = "mean.class")


ggplot(as.data.frame(ems), aes(subjectiveTimeFromLastIntactSec, mean.class,
                               color = condition, fill = condition)) + 
  geom_ribbon(aes(ymin = asymp.LCL, ymax = asymp.UCL),
              alpha = 0.2, color = NA) + 
  geom_line(size = 1) + 
  scale_y_continuous(limits = c(1, 4), oob = scales::squish) 




## =================== ANALYSIS 4 =================== 
"
In this analysis, we test whether there is a relationship between performance in the objective task (objectiveIsCorrect) 
and SOA (objectiveTimeFromLastIntactSec). 
** NOTE ** that the SOA here is different than the one in analysis 1. 
"

print("=================== ANALYSIS 4 =================== ")

## Models ------------------------

# Hypothesis
model2_h1 <- glmer(objectiveIsCorrect ~ objectiveTimeFromLastIntactSec + (objectiveTimeFromLastIntactSec | Subject),
                data = data, family = binomial())
print("-----------------------------model 2 summary-----------------------------")
print(summary(model2_h1))

# Null 
model2_h0 <- glmer(objectiveIsCorrect ~ (objectiveTimeFromLastIntactSec | Subject),
                   data = data, family = binomial())
print("-----------------------------model 2h0 summary-----------------------------")
print(summary(model2_h0))


## Comparisons ------------------------
"
We will compare the hypothesis model with each of the null models by calculating Bayes Factors based on the BIC approximation. 
"

print("-----------------------------model 2 comparisons: h1 v h0 -----------------------------")
# How much the alternative hypothesis is preferred over the null(i)?
print(bayesfactor_models(model2_h1, denominator = model2_h0))

sink()
