##### INTUIT analysis ##########
library("fixest"); library("rstatix"); library("lmerTest"); library("tidyverse")
library("ggplot2"); library("data.table"); library("marginaleffects")
library("margins"); library("modelsummary"); library("parameters"); library("readr")
library("lme4"); library("lfe"); library("this.path"); library("kableExtra"); 
library("knitr")

setwd(this.dir())

# Create a directory for LaTeX outputs
dir.create("paper_results/latex_tables", showWarnings = FALSE)

##### Summary data #####
df <- read_csv("paper_results/single_df.csv")

# Factor setup
df$model <- factor(df$model, levels = c("Human", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "o3-mini"))
df$condition <- factor(df$condition, levels = c("A", "B"))
df$inference_level <- factor(df$inference_level, levels = c("2", "0", "3"))
df$version <- factor(df$version, levels = c("A", "B"))
df$vignette_number <- factor(df$vignette_number)

# Base model
single_model <- lmer(
  accuracy ~ model + condition + inference_level + 
    model:condition +
    (1|id),
  data = df,
  control = lmerControl(optimizer = "bobyqa")
)

# Demands model
single_interaction_model <- lmer(
  accuracy ~ model + condition + inference_level + 
    model:condition +
    constitutional + functional + spatiotemporal + 
    beliefs + intentions + feelings +
    model:constitutional + model:functional + model:spatiotemporal +
    model:beliefs + model:intentions + model:feelings +
    (1|id),
  data = df
)

anova(single_model, single_interaction_model)

mean(df$accuracy, na.rm = TRUE)

#### double capability
df <- read_csv("paper_results/double_df.csv")
df$model <- factor(df$model, levels = c("Human", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "o3-mini"))
df$condition <- factor(df$condition, levels = c("A", "B", "C", "D"))
df$inference_level <- factor(df$inference_level, levels = c("2", "1", "3"))
df$version <- factor(df$version, levels = c("A", "B"))
df$vignette_number <- factor(df$vignette_number)

double_model <- lmer(
  accuracy ~ model + condition + inference_level + 
    model:condition +
    (1|id),
  data = df,
  control = lmerControl(optimizer = "bobyqa")
)

double_interaction_model <- lmer(
  accuracy ~ model + condition + inference_level + 
    model:condition +
    constitutional + functional + spatiotemporal + 
    beliefs + intentions + feelings +
    model:constitutional + model:functional + model:spatiotemporal +
    model:beliefs + model:intentions + model:feelings +
    (1|id),
  data = df
)

#### Perturbations
df <- read_csv("paper_results/perturbation_df.csv")

df$model <- factor(df$model, levels = c("gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"))
df$condition <- factor(df$condition, levels = c("A", "B"))
df$inference_level <- factor(df$inference_level, levels = c("2", "0", "3"))
df$version <- factor(df$version, levels = c("A", "B"))
df$vignette_number <- factor(df$vignette_number)
df$spacing <- factor(df$spacing, levels = c("Level 0", "Level 1", "Level 2", "Level 3"))
df$character <- factor(df$character, levels = c("Level 0", "Level 1", "Level 2", "Level 3"))
df$capitalisation <- factor(df$capitalisation, levels = c("Level 0", "Level 1", "Level 2", "Level 3"))
df$capability_type <- factor(df$capability_type, levels = c("single", "double"))

perturbation_model <- lmer(
  accuracy ~ model + condition + inference_level + 
    model:condition + capability_type +
    spacing + character + capitalisation +
    model:spacing + model:character + model:capitalisation + 
    (1|id),
  data = df,
  control = lmerControl(optimizer = "bobyqa")
)


# Export LaTeX summaries
modelsummary_config <- list(
  output = "latex",
  stars = TRUE,
  fmt = 3,
  statistic = "std.error"
)
options(modelsummary_factory_latex = "kableExtra")
options(modelsummary_format_numeric_latex = "plain")

latex_table <- modelsummary(
  list("Single Capability" = single_model, 
       "Double Capability" = double_model, 
       "Single with Demands" = single_interaction_model, 
       "Double with Demands" = double_interaction_model),
  output = "latex",
  fmt = 3,
  statistic = "std.error",
  stars = TRUE,
  longtable = TRUE
)
latex_table <- paste0("\\begingroup\\small\n", latex_table, "\n\\endgroup")
writeLines(latex_table, "paper_results/latex_tables/demand_summary.tex")

latex_table <- modelsummary(
  list("Perturbation Model" = perturbation_model),
  output = "latex",
  fmt = 3,
  statistic = "std.error",
  stars = TRUE,
  longtable = TRUE
)
latex_table <- paste0("\\begingroup\\small\n", latex_table, "\n\\endgroup")
writeLines(latex_table, "paper_results/latex_tables/perturbation_summary.tex")




