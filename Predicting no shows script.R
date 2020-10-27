# Title: Predicting medical appointment no shows
# Author: Jeremy Horne
# Date: 2020-10-23

# This script is based on the Kaggle 'Medical Appointment No Shows' dataset:
# See https://www.kaggle.com/joniarroba/noshowappointments for more information

# Load packages required for this script:
suppressPackageStartupMessages(library(kernlab))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyverse))

# Turn off scientific notation:
options(scipen="999")

# Set up file and folder locations:
Data_folder <- "C:\\Users\\horne\\OneDrive\\Documents\\R\\Patreon - No shows\\"
Filename <- "KaggleV2-May-2016.csv"

# Read data into R:
Raw_data <- read_csv(paste0(Data_folder,Filename))

# Re-code some of the text fields to numeric
Master_data <- Raw_data %>% mutate(Gender=str_replace(Gender,"M","1"),
                                      Gender=str_replace(Gender,"F","2"),
                                      `No-show`=str_replace(`No-show`,"No","0"),
                                      `No-show`=str_replace(`No-show`,"Yes","1"))

# Codify area to use for pattern recognition:
Areas <- Master_data %>% select(Neighbourhood) %>% distinct() %>% rowid_to_column("Area_ID")  

# Join area code back to the Master_data:
Master_data <- Master_data %>% left_join(Areas)

# Extract weekday, month and hour from date field
Master_data <- Master_data %>% mutate(BookingDay_DOW=wday(ScheduledDay, week_start = 1),
                                      BookingDay_Month=month(ScheduledDay),
                                      BookingDay_Day=day(ScheduledDay),
                                      BookingDay_Hour=hour(ScheduledDay),
                                      ApptDay_DOW=wday(AppointmentDay,week_start = 1),
                                      ApptDay_Month=month(AppointmentDay),
                                      ApptDay_Day=day(AppointmentDay),
                                      Booking_appt_day_diff=ceiling(difftime(AppointmentDay,
                                                                             ScheduledDay,units=c("days"))))

# This is where you may look to add in third party data sources
# e.g. COVID cases by day / weekly rolling average
# Public holidays
# Consumer spending index, etc.

# Now, let's assess the target variable - show / no show
Master_data %>% group_by(`No-show`) %>% summarise(percent=n()/nrow(Master_data))

# So 80% of people show up for their appointment and 20% don't
# This may make prediction difficult as we have four shows for every no show 
# The naive algorithm could therefore say 'I could do no work and get 80% of predictions correct'

# First, let's split into training and testing sets - do this based on appointment date:
Master_data %>% select(AppointmentDay) %>% summarise(max(AppointmentDay))
Master_data %>% select(AppointmentDay) %>% summarise(min(AppointmentDay))
Appointments_by_day <- Master_data %>% group_by(AppointmentDay) %>% summarise(Total=n()) %>% 
  arrange(AppointmentDay)

# Before splitting, ensure we only have numeric data for modelling:
Master_data_modelling <- Master_data %>% select(-c("ScheduledDay","Neighbourhood")) %>% 
  mutate_if(is.character,as.numeric)

# For the sake of building a quick model for this video, I'm going to sample just 10% of the data
# Normally of course - you would not do this!
Modelling_sample <- Master_data_modelling %>% slice_sample(prop=0.1)

# Lets use April and May as the training set, June as the testing set:
Master_train <- Modelling_sample %>% filter(AppointmentDay <= "2016-05-31") %>% select(-AppointmentDay)
Master_test <- Modelling_sample %>% filter(AppointmentDay > "2016-05-31") %>% select(-AppointmentDay)

# Build a standard model
# type="C-svc", kernel="vanilladot", C=1
set.seed(123)
Model_1 <- ksvm(`No-show` ~., data=Master_train, type="C-svc",kernel="vanilladot",C=1,prob.model=TRUE)

# Use the test set to make and check predictions:
Pred_1 <- predict(Model_1,Master_test,type="probabilities")

# The model has struggled due to the high proportion of "shows" in the training set
# Lets try and balance this some more - rather that 80:20, lets go 60:40
# You would probably try 70:30 first, but again, showing this an an example
Training_shows <- Master_train %>% filter(`No-show` == 0)
Training_no_shows <- Master_train %>% filter(`No-show` == 1)

# The 1,646 will make up 40% of your new training set - take a sample of the 'shows' as the other 60% ()
Train_4060 <- Training_shows %>% slice_sample(n=1.5*1646)

# Merge no-shows to create a single training set:
Train_4060 <- Train_4060 %>% bind_rows(Training_no_shows)

# Now use this to build a second model:
set.seed(123)
Model_2 <- ksvm(`No-show` ~., data=Train_4060, type="C-svc",kernel="vanilladot",C=1,prob.model=TRUE)

# Make some predictions with this:
Pred_2 <- predict(Model_2,Master_test,type="probabilities")

# Bring back in the actual values and analyse
Pred_2 <- Pred_2 %>% as.data.frame() %>% bind_cols(Master_test %>% select(`No-show`)) %>% 
  mutate(Band= cut(`1`, seq(0,1,0.1), right=FALSE))

# Analyse the preductions:
Pred_2_analysis <- Pred_2 %>% group_by(Band) %>% summarise(Total=n(),
                                                           Actual=sum(`No-show`),
                                                           Percent_correct=Actual/Total)

# From here, experiment further with splits (e.g.55/45 or 50/50) to determine best training composition
# Best split will allow you to identify individuals that have more than a 20% chance of not showing

# You can also amend the modelling parameters - for example on the 40/60 split
# The most obvious one is the kernel - vanilladot is really there for 'linear' problems
# The gaussian (or 'Radial basis') kernel is more suited to complex datasets - use "rbfdot" as kernel
set.seed(123)
Model_3 <- ksvm(`No-show` ~., data=Train_4060, type="C-svc",kernel="rbfdot",C=1,prob.model=TRUE)

# Predict and analyse
Pred_3 <- predict(Model_3,Master_test,type="probabilities")
Pred_3 <- Pred_3 %>% as.data.frame() %>% bind_cols(Master_test %>% select(`No-show`)) %>% 
  mutate(Band= cut(`1`, seq(0,1,0.1), right=FALSE))

Pred_3_analysis <- Pred_3 %>% group_by(Band) %>% summarise(Total=n(),
                                                           Actual=sum(`No-show`),
                                                           Percent_correct=Actual/Total)

# You can also change the cost parameter (C) - higher cost can help improve classification
# Higher cost is also computationally expensive, so a trade off
# Let's make the cost 10 times bigger
set.seed(123)
Model_4 <- ksvm(`No-show` ~., data=Train_4060, type="C-svc",kernel="rbfdot",C=10,prob.model=TRUE)

# Predict and analyse:
Pred_4 <- predict(Model_4,Master_test,type="probabilities")
Pred_4 <- Pred_4 %>% as.data.frame() %>% bind_cols(Master_test %>% select(`No-show`)) %>% 
  mutate(Band= cut(`1`, seq(0,1,0.1), right=FALSE))

Pred_4_analysis <- Pred_4 %>% group_by(Band) %>% summarise(Total=n(),
                                                           Actual=sum(`No-show`),
                                                           Percent_correct=Actual/Total)

# The type can also be changed - but make sure you stick to the type of problem you are trying to solve
# svc for classification and svr for regression
# More information can be found at https://cran.r-project.org/web/packages/kernlab/kernlab.pdf
set.seed(123)
Model_5 <- ksvm(`No-show` ~., data=Train_4060, type="nu-svc",kernel="vanilladot",C=1,prob.model=TRUE)

# Predict and analyse:
Pred_5 <- predict(Model_5,Master_test,type="probabilities")
Pred_5 <- Pred_5 %>% as.data.frame() %>% bind_cols(Master_test %>% select(`No-show`)) %>% 
  mutate(Band= cut(`1`, seq(0,1,0.1), right=FALSE))

Pred_5_analysis <- Pred_5 %>% group_by(Band) %>% summarise(Total=n(),
                                                           Actual=sum(`No-show`),
                                                           Percent_correct=Actual/Total)
