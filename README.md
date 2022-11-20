# Multi-trial Inattentional Blindness in Virtual Reality : Analysis

This code handles data that is the output of the VRIB paradigm (see https://osf.io/6jyqx).
In our paradigm (a VR multi-trial IB paradigm, henceforth VRIB), subjects are immersed in an urban environment while engaged in a task of following a target bee out of a group of three bees. 
Meanwhile, critical stimuli (taken from IAPS database: Bradley and Lang, 2007) are embedded in the city street. 
Each trial contains ten presentations of a single critical stimulus; in three of them, the stimulus image is intact, and in seven, a scrambled version of it appears. 
At the end of each trial, subjects are asked to select the target bee, rate the critical stimulus’ visibility on the Perceptual Awareness Scale (PAS; Ramsøy and Overgaard, 2004, a four-item scale where: 1=no experience, 2=brief glimpse, 3=almost clear experience, 4=clear experience) and select the stimulus image out of an array of four images (a 4-alternative forced choice task; 4AFC, an objective measure of awareness). 

In the first phase of the experiment (trials 1-40), the trials are novel, and participants gain/lose money based on making the correct bee selection. The bees' speed is affected by factors such as subjects' answers and the number of clues they took, such that the speed in the following trial increase or decreases based on their performance. 
In the second phase of the experiment (trials 41-50), the trials re-play randomly selected trials from the first phase; subject's are no longer supposed to play to maximize their gain, and the bees' speed is not affected by their performance (as the next trials is a replay of a previously played trial). Participants were instructed to look at the bus stops at the sides of the road, and ignore the bees. Notably, they were still asked all three questions at the end of each trial (select the bee, PAS rating, 4AFC). 

The experiment's output is comprised of multiple txt files; some are outputted after each trial, while others are only updated and include information about all trials. 
Thus, the custom-made Python data processing module aggregates each participant's collection of txt files - and puts it into readable dataframes that are easily aggregated and analyzed. The R script is where the linear modelling is performed. The rest of the analyses are done on JASP based on other csv outputs. 

## Python 

### Analysis manager
The main function in this code is the analysis manager, which calls all the relevant processing modules. First, it calls the parsing module, then the exclusion module to exclude subjects based on pre-defined criteria, and then it calls the two processing modules: the behavioral analysis and the gaze analysis. 
Gaze data was collected via VIVE PRO EYE's eye tracker and logged via an internal logging system. 

### Data parsing
The parse_data_files module loads and parses the VRIB data, managed by the main method "extract_subject_data" which is called by the analysis manager module. As each subject produces many output files during the experiment, this module handles their integration into a single data point that includes all the relevant subject-level information in one place. 

### Data exclusion
The exclusion_criteria module manages everything related to participant data exclusion, based on pre-registered requirements regarding their behavior. The main method here is beh_exclusion, which is called by the analysis_manager module. 

### Behavioral data analysis preparation
The beh_analysis module manages everything related to the processing of behavioral data towards analysis. 
Note that the statistical analyses themselves (linear mixed models, t-tests etc) are not done here; the goal of this module is to output summary data and plots, as well as aggregate data into a group dataframe that will be later analyzed with R (for linear models) and JASP (for t-tests). 

### Gaze data analysis preparation
The gaze_analysis module manages everything related to the processing of eye-tracking data towards analysis. 
Note that the statistical analyses themselves (linear mixed models, t-tests etc) are not done here; the goal of this module is to output summary data and plots, as well as aggregate data into a group dataframe that will be later analyzed with R (for linear models) and JASP (for t-tests). 

### Plotting
The plotter module is called from all modules that do any plotting. It contains a function per plot type. 

### Peripheral analysis (depracated)
Two pilot experiments preceeded this preregistered experiment, which is the one to be preprocessed by the current code. In the pilots, participants wore an [empatica bracelet](https://www.empatica.com/) which took peripheral measures such as temperature and heart-rate. As this data was not measured in the preregistered experiment, codes related to this peripheral analysis are commented out, and marked as depracated.

## R
Once the data is processed, the linear model analysis is done with R using the vrib_analysis script, run on the "avg_gaze_per_pas_long_intact.csv" file which is outputted by the et_analysis method in the gaze_analysis module. It combines both gaze and behavior data that are needed for the R modelling. 

#### Author
[Rony Hirschhorn](https://github.com/RonyHirsch/)