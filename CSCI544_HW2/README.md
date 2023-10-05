To execute the code and reproduce the results perform following steps

1. Install the required libraries from requirements.txt. (Only numpy library needs to be installed)
2. Set the parameters used in code line::425 in code (threshold_unknown, is_dev_run,use_pseudo_words,init_value, use_log_likelihood)
    The highest accuracy is obtained with following parameters (they are set by default)
    
    a. threshold_unknown = 2           ## Threshold, any word with less than N occurrences in train will be considered as unknown word
    
    b. is_dev_run = True               ## Flag to check the accuracy on dev dataset, setting this false will just run the code on test dataset, reducing runtime of the program
    
    c. use_pseudo_words = True         ## Flag to use the pseudo words function to convert unknown words
    
    d. init_value = 0                  ## Setting initial value of vocab word initialization
    
    e. use_log_likelihood = True       ## Set if you need to use log likelihood while calculating probabilities

3. Run the python file with following command on command line - 
        python <path_to_file> <path_to_data_folder> <path_to_verification_folder>

    e.g. - python .\Homework2_HMM.py .\data .\verification
    e.g. - python CSCI544_HW2\Homework2_HMM.py CSCI544_HW2\data CSCI544_HW2\verification 

4. The accuracy for dev dataset and answers to questions will be printed on console and the txt and json files will be output to out folder in verification folder.

5. Total Execution time 
    With prediction of dev and test labels and calculating accuracy - 16 mins
    With prediction of just test labels - 7 mins