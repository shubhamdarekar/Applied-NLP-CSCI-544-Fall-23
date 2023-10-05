"""
Shubham Sanjay Darekar
CSCI 544 - Applied Natural language Processing

Average runtime of this script - 16 mins

"""


import json
import numpy as np
import json
import time
import re
import sys


def read_parse_data(data_path:str):
    """
    Reads and parses json data from a json file
    Inputs - 
        data_path:str   : path to json file

    Outputs - 
        data: parsed list or dict object
    """
    with open(data_path) as f:
        data = json.load(f)

    return data




def vocab_creation(train_data: list, vocab_path:str, use_pseudo_words:bool, threshold:int = 3, init_value:int = 0):
    """
    Reads and iterate through the train dataset and outputs vocabulary to vocab.txt
    Inputs - 
        train_data:list         : Parsed json object of train data with keys labels and sentences
        vocab_path:str          : absolute path to output file (txt file)
        use_pseudo_words:bool   : Flag stating if pseudo word implemtion should be considered
        threshold:int           : Any word with less than N occurances in train will be considered as unknown word (optional, def: 3)
        init_value:int          : Initial value of vocab word initialization used for smoothening (optional, def: 0)
    Output - 
        dict_vocab_sentences    : Dictionary containing unique words and occurance count
        dict_vocab_pos          : Dictionary containing unique pos tags and occurance count
        train_data              : Updated train dataset using the parameters defined (pesudowords and thresholds)
    """

    
    dict_vocab_sentences = dict()   # holds the actual words and their occurances
    dict_vocab_pos = dict()         # holds the Part of speech tags and their occurances

    
    ## Reading all the data from train dataset and setting occurances
    for row in train_data:
        for word in row['sentence']:
            dict_vocab_sentences[word] = dict_vocab_sentences.get(word,init_value) + 1
        
        for pos_tag in row['labels']:
            dict_vocab_pos[pos_tag] = dict_vocab_pos.get(pos_tag,init_value) + 1
    dict_vocab_pos['<end>'] = len(train_data) ## Adding end charcter.   

    
    ## Updating the less frequent words in the dictionary for intermidiate use
    dict_vocab_sentences_updated = dict()
    for word in dict_vocab_sentences.keys():
        freq = dict_vocab_sentences[word]
        if freq < threshold:
            word = psudo_words_convert(word,use_pseudo_words) ## Converting on basis of pseudo words function
        dict_vocab_sentences_updated[word] = dict_vocab_sentences_updated.get(word,0) + freq
    

    ## Sorting the dictionary 
    # ## referred to this post for fastest method to sort dict- https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    dict_vocab_sentences = dict(sorted(dict_vocab_sentences_updated.items(), key=lambda x:x[1],reverse=True))
    dict_vocab_pos = sorted(dict_vocab_pos.items(), key=lambda x:x[1],reverse=True)

   
    ## Creating the string for vocab.txt
    text_vocab = ""
    counter = 0
    for key in dict_vocab_sentences.keys():
        text_vocab += key + "\t"+ str(counter) +"\t" + str(dict_vocab_sentences[key]) +"\n"
        counter+=1
    with open(vocab_path,'w') as v:
        v.write(text_vocab[:-1])
    v.close()


    ## Replacing the less frequent words in the `train datasets`` according to parameters
    for row_num in range(len(train_data)):
        for word_num in range(len(train_data[row_num]['sentence'])):
            if train_data[row_num]['sentence'][word_num] in dict_vocab_sentences:
                pass
            else:   
                train_data[row_num]['sentence'][word_num] = psudo_words_convert(train_data[row_num]['sentence'][word_num],use_pseudo_words)


    return dict_vocab_sentences, dict(dict_vocab_pos), train_data



def model_learning(train_data:list, dict_vocab_pos:dict, hmm_path:str):
    """
    Method to calculate initial probabilities, emission probabilities and transition probabilities
    Writes the dictionaries to hmm.json file.
    Inputs - 
    train_data:list         : Processed training dataset
    dict_vocab_pos:dict     : list of pos tags and their occurances
    hmm_path:str            : Path to output json file
    
    Outputs -
    emission        : Emission probabilities - key - (tag, word)
    transition      : Transition probabilites - key - (tag, next_tag)
    initial_pi      : Initial probabilities - key - (tag)
    """
    
    ## Creating empty dictionaries
    emission = dict()
    transition = dict()
    initial_pi = dict()


    ## Iterating through each sentence of the training corpus
    for row in train_data:
        
        ## Iterating through each word of the selected sentence
        for i in range(len(row['labels'])):
            
            cur_s = row['labels'][i]
            cur_x = row['sentence'][i]

            emission[(cur_s,cur_x)] = emission.get((cur_s,cur_x),0)+1 ## counting emissions for (tag, word)

            if i == 0:
                initial_pi[cur_s] = initial_pi.get(cur_s,0)+1  ## counting initial occurances for tag
            else:
                prev_s = row['labels'][i-1]
                transition[(prev_s,cur_s)] = transition.get((prev_s,cur_s),0)+1     ## counting emissions for (tag1, tag2)

        ## Adding end tag and word token for all the sentences in the corpus         
        transition[(cur_s,'<end>')] = transition.get((cur_s,'<end>'),0) + 1
        emission[('<end>','<end>')] = emission.get(('<end>','<end>'),0) + 1


    ## Creating new dictionaries to change the keys to string (this is only done to create hmm.json)
    emission_write_version = dict()
    transition_write_version = dict()
    initial_write_version = dict()


    ## Dividing by total occurances
    for e in emission.keys():
        emission[e] = emission[e] / dict_vocab_pos.get(e[0],0)
        emission_write_version[str(e)] = emission[e]

    for t in transition.keys():
        transition[t] = transition[t] / dict_vocab_pos.get(t[0],0)
        transition_write_version[str(t)] = transition[t]
    
    for p in initial_pi.keys():
        initial_pi[p] = initial_pi[p] / len(train_data)
        initial_write_version[str(p)] = initial_pi[p]


    ## Writing to hmm.json file
    with open(hmm_path,'w') as h:
        json.dump({'initial':initial_write_version,'transition':transition_write_version,'emission':emission_write_version},h,indent=4)
    h.close()

    return emission, transition, initial_pi



def greedy_hmm_decoding(emission: dict, transition: dict, initial_pi: dict, dict_vocab_sentences:dict,test_data: list,dict_vocab_pos: dict,greedy_out_path:str, use_pseudo_words:bool, use_log_likelihood:bool,eps:float):
    """
    Greedy decoding algorithm. 
    Writes the greedy.json file with predicted labels
    
    Inputs -
        emission: dict              : Emission probabilities - key - (tag, word)
        transition: dict            : Transition probabilites - key - (tag, next_tag)
        initial_pi: dict            : Initial probabilities - key - (tag)
        dict_vocab_sentences:dict   : Dictionary containing unique words and occurance count
        test_data: list             : Processed training dataset
        dict_vocab_pos: dict        : list of pos tags and their occurances
        greedy_out_path:str         : Path to greedy.json output file
        use_pseudo_words:bool       : Flag stating if pseudo word implemtion should be considered
        use_log_likelihood:bool     : Flag stating if log likelihood implemtion should be considered
        eps:float                   : Small number added to avoid math error in log
    
    Outputs -
        test_data       : predicted labeled test dataset

    """
    ## getting set of known words
    vocab_list = set(dict_vocab_sentences.keys())

    ## iterating through each sentence of testing data
    for row_num in range(len(test_data)):
        
        ## adding end token to the sentence
        sentence = test_data[row_num]['sentence'] + ['<end>']
        
        pos_set = [""]*len(sentence)  ## initializing set of pos tags
        
        ## iterating through each word of selected sentence
        for i in range(len(sentence)):
            max_prob = float('-inf')    ## Setting to -inf for log likelihood implementation
            label = 'NN'                ## Assigning the most common label to all the unknown words after comma 
            
            cur_word = sentence[i]

            ## Updating the unknown words
            if cur_word in vocab_list:
                pass ## avoiding use of (not in) to reduce runtime
            else:
                cur_word = psudo_words_convert(cur_word,use_pseudo_words)
            

            ## iterating through each tag for each word
            for pos_label in dict_vocab_pos.keys():
                
                if i == 0: ## initial word pass ( checking the initial probabilities for the first word)
                    ## if else to check if log likelihood implementation is to be used
                    if use_log_likelihood:
                        p = np.log(initial_pi.get(pos_label,0 ) + eps) + np.log(emission.get((pos_label,cur_word),0) + eps)
                    else:
                        p = initial_pi.get(pos_label,0 ) * emission.get((pos_label,cur_word),0)
                    
                    ## replacing the max probability
                    if p > max_prob:
                        label = pos_label
                        max_prob = p
                
                else:
                    ## if else to check if log likelihood implementation is to be used
                    if use_log_likelihood:
                        p = np.log(transition.get((pos_set[i-1],pos_label),0) + eps) + np.log(emission.get((pos_label,cur_word),0) + eps)
                    else:
                        p = transition.get((pos_set[i-1],pos_label),0) * emission.get((pos_label,cur_word),0)
                    
                    ## replacing the max probability
                    if p > max_prob:
                        label = pos_label
                        max_prob = p
            pos_set[i] = label

        
        test_data[row_num]['labels'] = pos_set[:-1] ## renoved the end tag from the predictions

    ## Writing the predictions to greedy.json
    with open(greedy_out_path,'w') as h:
        json.dump(test_data,h)
    h.close()

    return test_data




def viterbi_hmm_decoding(emission:dict ,transition:dict , initial_pi: dict, dict_vocab_sentences:dict, test_data:list ,dict_vocab_pos:dict,viterbi_out_path:str , use_pseudo_words:bool, use_log_likelihood:bool,eps:float):
    """
    Viterbi decoding algorithm. 
    Writes the viterbi.json file with predicted labels
    
    Input - 
        emission: dict              : Emission probabilities - key - (tag, word)
        transition: dict            : Transition probabilites - key - (tag, next_tag)
        initial_pi: dict            : Initial probabilities - key - (tag)
        dict_vocab_sentences:dict   : Dictionary containing unique words and occurance count
        test_data: list             : Processed training dataset
        dict_vocab_pos:dict         : list of pos tags and their occurances
        viterbi_out_path:str        : path to viterbi.json output file
        use_pseudo_words:bool       : Flag stating if pseudo word implemtion should be considered
        use_log_likelihood:bool     : Flag stating if log likelihood implemtion should be considered
        eps:float                   : Small number added to avoid math error in log
    Output
        test_data       : predicted labeled test dataset
    """
    
    ## getting set of all tags
    tags_set = list(dict_vocab_pos.keys())
    total_tags = len(tags_set)
    
    ## getting set of known words
    vocab_list = set(dict_vocab_sentences.keys()) 
    

    ## iterating through each sentence in output
    for row_num in range(len(test_data)):
        sentence = test_data[row_num]['sentence'] + ['<end>']
        total_words = len(sentence)

        ### Calculating the probabilities
        ## initializing viterbi probability matrix and storage matrix
        viterbi_prob = np.zeros((total_words,total_tags))
        prev_storage = np.zeros((total_words,total_tags), dtype=int)

        tag_set_final = ['NN'] * total_words ## setting NN as default as it is the most common label (works only with threshold <1)

        ## iterating through the words in sentence
        for word_num in range(total_words):
            word = sentence[word_num]
            
            
            ## Updating the unknown words
            if word in vocab_list:
                pass ## avoiding use of (not in) to reduce runtime
            else:
                word = psudo_words_convert(word,use_pseudo_words)

            
            for tag_num in range(total_tags):
                if word_num == 0: ## initial word pass ( checking the initial probabilities for the first word)
                    ## if else to check if log likelihood implementation is to be used
                    if use_log_likelihood:
                        viterbi_prob[word_num][tag_num] = np.log(initial_pi.get(tags_set[tag_num],0)+ eps) + np.log(emission.get((tags_set[tag_num],word),0) + eps)
                    else:
                        viterbi_prob[word_num][tag_num] = initial_pi.get(tags_set[tag_num],0) * emission.get((tags_set[tag_num],word),0)
                    
                else:
                    ## if else to check if log likelihood implementation is to be used
                    if use_log_likelihood:
                        p = np.add(viterbi_prob[word_num-1], 
                                   [np.log(transition.get((z,tags_set[tag_num]),0)+eps) + 
                                   np.log(emission.get((tags_set[tag_num],word),0) +eps) 
                                   for z in tags_set])
                    else:
                        p = np.multiply(viterbi_prob[word_num-1], 
                                            [transition.get((z,tags_set[tag_num]),0) * 
                                             emission.get((tags_set[tag_num],word),0) 
                                             for z in tags_set])
                    
                    ## storing maximum index of maximum probability in prev_storage and maximum probability in viterbi_prob
                    ## Reference for finding maximum index - https://stackoverflow.com/questions/11530799/python-finding-index-of-maximum-in-list
                    prev_storage[word_num][tag_num] = p.argmax()
                    viterbi_prob[word_num][tag_num] = p.max()

        ### Backtracking

        ## Finding max probability in last word's all tags probability
        max_prob_index= viterbi_prob[-1].argmax()
        tag_set_final[-1] = tags_set[max_prob_index]
        
        ## backtracking from storage matrix
        for index_neg in range(total_words-1,0,-1):
            max_prob_index = prev_storage[index_neg][max_prob_index]
            tag_set_final[index_neg-1] = tags_set[max_prob_index]
        
        test_data[row_num]['labels'] = tag_set_final[:-1] ## ignoring the end token

    ## writing the labels to output path and generating viterbi.json
    with open(viterbi_out_path,'w') as h:
        json.dump(test_data,h)
    h.close()

    return test_data



def test_accurary(prediction:str ,actual:str):
    """
    Comparing tags and returning accuracy
    Inputs - 
        prediction: str     : path to predicted labels
        actual: str         : path to actual labels
    
    Outputs - 
        accuracy:  accuray of the prediction algorithm
    """
    data_prediction = read_parse_data(prediction)
    data_actual = read_parse_data(actual)

    total_predictions = 0
    correct_prediction = 0

    for row_num in range(len(data_prediction)):
        for i in range(len(data_prediction[row_num]['labels'])):
            if data_prediction[row_num]['labels'][i] == data_actual[row_num]['labels'][i]:
                correct_prediction+=1
            total_predictions+=1
    
    return correct_prediction/total_predictions



def psudo_words_convert(word:str, use_pseudo_words: bool):
    """
    Converts the unknown words into pseudo words or assigns <unk> tag
    ## Reference - http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf

    Inputs - 
    word:str                : Input word
    use_pseudo_words: bool  : Flag stating if pseudo word implemtion should be considered

    Outputs -
    string  :   Pseudo word or <unk> tag 

    """
    if not use_pseudo_words:
        return '<unk>'
    else:
        if re.match(r'^[A-Z][a-z]*$',word):
            return 'init_cap'
        elif re.match(r'^[0-9]{4}$',word):
            return 'four_digits'
        elif re.match(r'^.*\.$',word):
            return 'words_ending_with_period'
        elif re.match(r'^[A-Z]*$',word):
            return 'all_caps'
        elif re.match(r'^[a-z]*$',word):
            return 'lowercase'
        elif re.match(r'^[0-9]*\.[0-9]*$',word):
            return 'number_and_period'
        elif re.match(r'^[0-9]*\,[0-9]*\.[0-9]*$',word):
            return 'number_and_period_and_comma'
        else:
            return '<unk>'


if __name__ ==  "__main__":

    threshold_unknown = 2           ## Threshold, any word with less than N occurrences in train will be considered as unknown word
    is_dev_run = True               ## Flag to check the accuracy on dev dataset
    use_pseudo_words = True         ## Flag to use the pseudo words function to convert unknown words
    init_value = 0                  ## Setting initial value of vocab word initialization
    use_log_likelihood = True       ## Set if you need to use log likelihood while calculating probabilities
    eps = 1e-300                    ## using small epsilon value for log likelihood 


    ## Getting file paths from command line
    try:
        cmd_arguments = sys.argv
        data_path = ""
        verification_path = ""
        if len(cmd_arguments) > 1:
            data_path = cmd_arguments[1]
            verification_path = cmd_arguments[2]
    except:
        print("Please provide the path to the data folder and verification folder in the command, check readme file for format")
        exit()
    

    ## Settting the paths of all the input files
    train_data_path = data_path + '\\train.json'
    test_data_path = data_path + '\\test.json'
    dev_data_path = data_path + '\\dev.json'

    ## Setting paths of all the output files
    vocab_out_path = verification_path + '\\out\\vocab.txt'
    hmm_out_path = verification_path + '\\out\\hmm.json'
    greedy_out_path = verification_path + '\\out\\greedy.json'
    viterbi_out_path = verification_path + '\\out\\viterbi.json'
    greedy_out_path_dev = verification_path + '\\out\\greedy_dev.json'
    viterbi_out_path_dev = verification_path + '\\out\\viterbi_dev.json'
    

    
    ## VOCAB CREATION
    train_data = read_parse_data(train_data_path)
    dict_vocab_sentences, dict_vocab_pos, train_data = vocab_creation(train_data, vocab_out_path,use_pseudo_words, threshold_unknown, init_value)


    print("\nWhat threshold value did you choose for identifying unknown words for replacement?")
    print("--->>>>> "+ str(threshold_unknown))

    print("\nWhat is the overall size of your vocabulary?")
    print("--->>>>> "+ str(len(dict_vocab_sentences))+" (Includes 8 pseudo words)")

    print("\nHow many times does the special token ”< unk >” occur following the replacement process?")
    print("""---->>>>> \nFollowing tokens/ pseudo words were used in replacement process. The corresponding occurances are mentioned""")
    pseudo_words_set = ['<unk>','init_cap','four_digits','words_ending_with_period','all_caps','lowercase', 'number_and_period','number_and_period_and_comma']
    total_unk = 0
    for pseudo in pseudo_words_set:
        print(pseudo + " :"+str(dict_vocab_sentences.get(pseudo,0)))
        total_unk += dict_vocab_sentences.get(pseudo,0)
    print("\nTotal replaced: "+str(total_unk))


    ## MODEL LEARNING
    emission, transition, initial_pi = model_learning(train_data,dict_vocab_pos, hmm_out_path)

    print("\n\nHow many transition and emission parameters in your HMM?")
    print("--->>>>>")
    print("Initial_pi : "+str(len(initial_pi)))
    print("Emissions : "+str(len(emission)))
    print("Transition : "+str(len(transition)))
    


    ## GREEEDY DECODING
    if is_dev_run:
        dev_data = read_parse_data(dev_data_path)
        time_stamp_1 = time.time()
    
        
        test_data_labeled_greedy = greedy_hmm_decoding(emission,transition, initial_pi, dict_vocab_sentences, dev_data ,dict_vocab_pos,greedy_out_path_dev, use_pseudo_words, use_log_likelihood, eps)
        time_stamp_2 = time.time()
        print("\nGreedy Decoding for dev Executed in " + str(time_stamp_2 - time_stamp_1))
        
        print("\nWhat is the accuracy on the dev data?")
        print("--->>>>> Greedy dev Accuracy: " +str(round(test_accurary(greedy_out_path_dev,dev_data_path)*100,3))+" %")
    
    test_data = read_parse_data(test_data_path)
    time_stamp_3 = time.time()
    test_data_labeled_greedy = greedy_hmm_decoding(emission,transition, initial_pi, dict_vocab_sentences, test_data,dict_vocab_pos,greedy_out_path, use_pseudo_words, use_log_likelihood, eps)
    time_stamp_4 = time.time()
    print("\nGreedy Decoding for test Executed in " + str(time_stamp_4 - time_stamp_3))

    

    ## VITERBI DECODING
    if is_dev_run:

        test_data_labeled_viterbi = viterbi_hmm_decoding(emission,transition, initial_pi, dict_vocab_sentences, dev_data,dict_vocab_pos,viterbi_out_path_dev,use_pseudo_words, use_log_likelihood, eps)
        time_stamp_5 = time.time()
        print("\nViterbi Decoding for dev Executed in " + str(time_stamp_5 - time_stamp_4))
        print("\n\nWhat is the accuracy on the dev data?")
        print("--->>>>> Viterbi Accuracy: " +str(round(test_accurary(viterbi_out_path_dev,dev_data_path)*100,3)) +" %")
    
    
    test_data_labeled_viterbi = viterbi_hmm_decoding(emission,transition, initial_pi, dict_vocab_sentences, test_data,dict_vocab_pos,viterbi_out_path,use_pseudo_words, use_log_likelihood, eps)
    time_stamp_6 = time.time()
    print("\nViterbi Decoding for test Executed in " + str(time_stamp_6 - time_stamp_5))