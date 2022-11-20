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


## =================== ANALYSIS 1 =================== 
"
In this analysis, we test whether PAS (subjectiveAwareness) is different across conditions (condition), 
as well as test whether there is a relationship between PAS (subjectiveAwareness) and SOA (subjectiveTimeFromLastIntactSec). 
"

## Models ------------------------

# Hypothesis
model1_h1 <- clmm(subjectiveAwareness ~ condition + subjectiveTimeFromLastIntactSec + (condition + subjectiveTimeFromLastIntactSec | Subject),
              data = data)
summary(model1_h1)

# Null i
model1_h0i <- clmm(subjectiveAwareness ~  subjectiveTimeFromLastIntactSec + (condition + subjectiveTimeFromLastIntactSec | Subject),
              data = data)
summary(model1_h0i)

# Null ii 
model1_h0ii <- clmm(subjectiveAwareness ~  condition + (condition + subjectiveTimeFromLastIntactSec | Subject),
                 data = data)
summary(model1_h0ii)

## Comparisons ------------------------
"
We will compare the hypothesis model with each of the null models by calculating Bayes Factors based on the BIC approximation. 
"

# How much the alternative hypothesis is preferred over the null(i)?
bayesfactor_models(model1_h1, denominator = model1_h0i)  # this calculates the BIC for both models, then turns them into BFs

# How much the alternative hypothesis is preferred over the null(ii)?
bayesfactor_models(model1_h1, denominator = model1_h0ii)

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

## Models ------------------------

# Hypothesis
model2_h1 <- glmer(objectiveIsCorrect ~ objectiveTimeFromLastIntactSec + (objectiveTimeFromLastIntactSec | Subject),
                data = data, family = binomial())
summary(model2_h1)

# Null 
model2_h0 <- glmer(objectiveIsCorrect ~ (objectiveTimeFromLastIntactSec | Subject),
                   data = data, family = binomial())
summary(model2_h0)

## Comparisons ------------------------
"
We will compare the hypothesis model with each of the null models by calculating Bayes Factors based on the BIC approximation. 
"

# How much the alternative hypothesis is preferred over the null(i)?
bayesfactor_models(model2_h1, denominator = model2_h0)
