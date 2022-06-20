# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

"""
Given,
sentence: str - e.g Tesla surged 10% today!
labels: list[list[int,int,str]] - e.g [[0,5,"POS"]]
returns the aspects: list[str] - e.g ["Tesla"]
"""
def get_aspect_words(sentence:str, labels:list):
    return [sentence[label[0]:label[1]] for label in labels]
"""
Given,
sentence: str - e.g Tesla surged 10% today!
labels: list[list[int,int,str]] - e.g [[0,5,"POS"]]
aspect_identifier:str - e.g INVULNERABLE_ASPECT (single token not in lexicon)
returns the sentence with the aspects replaced with "INVULNERABLE_ASPECT": str
e.g INVULNERABLE_ASPECT surged 10 % today !
"""
def replace_aspect_with_aspect_identifier(sentence:str, labels:list, aspect_identifier: str):
    for label in labels[::-1]:   # MUST loop from back
        sentence = f"{sentence[:label[0]]} {aspect_identifier} {sentence[label[1]:]}"
    return sentence
"""
Augmented training samples refer to the augmented sentence AND the new labels
Given:
1. aug_words: list[str] - e.g ["INVULNERABLE_ASPECT", "increased", "10", "%", "today", "!"]
2. labels: list[list] - e.g [[0,18,POS]]
3. aspects: list[str] - ["NVIDIA Corporation"]
4. aspect_identifier: str - e.g INVULNERABLE_ASPECT
returns:
1. aug_sentence:str - NVIDIA Corporation increased 10% today!
2. new_labels:list[list] - e.g [[0,18,POS]]
"""
def get_augmented_sample(aug_words, labels, aspects, aspect_identifier):

    aspect_index = 0
    new_labels = []
    for i,word in enumerate(aug_words):
        if word == aspect_identifier:
            aspect_len = len(aspects[aspect_index])
            pre_length = len(" ".join(aug_words[:i]))+1 if i>0 else len(" ".join(aug_words[:i]))
            aug_words[i] = aspects[aspect_index]            # add back original aspect word
            new_labels.append([pre_length,pre_length+aspect_len,labels[aspect_index][2]]) # get new labels
            aspect_index+=1
    aug_sentence = " ".join(aug_words)
    return aug_sentence, new_labels

"""
Given:
word: str - increased
return synonyms: list[str] - e.g  ["surged", "gained", "boosted"]
"""
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms) 


########################################################################
# Synonym replacement
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize

"""
performs synonym_replacement on n words(exluding the aspect_identifier, punctuations and stop words)
words: list[str] e.g ["INVULNERABLE_ASPECT", "increased", "10", "%", "today", "!"]
n: int 
aspect_identifier: str - must be a single token not in nltk dictionary e.g INVULNERABLE_ASPECT
"""
def synonym_replacement(words, n, aspect_identifier):   
        
    new_words = words.copy()
    
    # Pool of words to replace must not be stop words, punctuations or aspects
    random_word_list = list(set([word for word in words if word not in stop_words and word.isalpha() and word!=aspect_identifier]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]    # this might make multiple changes
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    return new_words


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

"""
Given: 
words: list[str] e.g ["INVULNERABLE_ASPECT", "increased", "10", "%", "today", "!"]
p: float e.g 0.1
aspect_identifier: str - must be a single token not in nltk dictionary e.g INVULNERABLE_ASPECT
delete tokens in "words" with probability p, excluding aspect_identifier
"""
def random_deletion(words, p, aspect_identifier):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        if word==aspect_identifier:
            new_words.append(word)
        else:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
########################################################################

"""
Given: 
words: list[str] e.g ["INVULNERABLE_ASPECT", "increased", "10", "%", "today", "!"]
n: int e.g 1
aspect_identifier: str - must be a single token not in nltk dictionary e.g INVULNERABLE_ASPECT
swap two words n times in "words", excluding aspect_identifier
"""
def random_swap(words, n, aspect_identifier):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, aspect_identifier)
    return new_words
def swap_word(new_words, aspect_identifier):
    # choose first random word to swap (must not be the aspect_identifier)
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    while new_words[random_idx_1] == aspect_identifier:   
        random_idx_1 = random.randint(0, len(new_words)-1)
    # choose second word to swap (must not be the first word or aspect_identifier)
    counter = 0
    while random_idx_2 == random_idx_1 or new_words[random_idx_2] == aspect_identifier:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 5:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
########################################################################

"""
Randomly insert n words into the sentence
"""
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, labels, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, aspect_identifier = "INVULNERABLE_ASPECT"):
    
    # OUTPUT 
    augmented_sentences = [] # list[str]
    label_list = []          # updated labels
    
    # labels MUST be sorted 
    labels.sort()                                                                             
    # list of aspect words in sequence to be added back later
    aspects = get_aspect_words(sentence,labels)                                               # e.g ['NVIDIA Corporation']
    # replace aspects with invulnerable token (aspect_identifier) so they're not replaced
    tmp_sentence = replace_aspect_with_aspect_identifier(sentence,labels,aspect_identifier)   # e.g INVULNERABLE_ASPECT surged 10% today!
    words = word_tokenize(tmp_sentence)                                # e.g ["INVULNERABLE_ASPECT", "surged", "10", "%", "today", "!"]  
    

    # number of words in the sentence, excluding punctuations
    num_words = sum([1 for word in words if word.isalpha()])
    # number of augmented samples to generated per eda technique
    num_new_per_technique = int(num_aug/4)+1

    #SYNONYM REPLACEMENT
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            # e.g ["INVULNERABLE_ASPECT", "increased", "10", "%", "today", "!"]
            aug_words = synonym_replacement(words, n_sr, aspect_identifier)
            # e.g NVIDIA Corporation increased 10% today!, [[0,18,POS]]
            aug_sentence, new_labels = get_augmented_sample(aug_words, labels, aspects, aspect_identifier)
            # update outputs
            augmented_sentences.append(aug_sentence)
            label_list.append(new_labels)

    # RANDOM INSERTION
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            
            # e.g ["INVULNERABLE_ASPECT", "surged", "increased", "10", "%", "today", "!"]
            aug_words = random_insertion(words, n_ri)
            # e.g NVIDIA Corporation surged increased 10% today!, [[0,18,POS]]
            aug_sentence, new_labels = get_augmented_sample(aug_words, labels, aspects, aspect_identifier)
            # update outputs
            augmented_sentences.append(aug_sentence)
            label_list.append(new_labels)

    # RANDOM SWAP
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            aug_words = random_swap(words, n_rs, aspect_identifier)
            aug_sentence, new_labels = get_augmented_sample(aug_words, labels, aspects, aspect_identifier)
            # update outputs
            augmented_sentences.append(aug_sentence)
            label_list.append(new_labels)
            
    # RANDOM DELETION
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            
            aug_words = random_deletion(words, p_rd, aspect_identifier)
            aug_sentence, new_labels = get_augmented_sample(aug_words, labels, aspects, aspect_identifier)
            # update outputs
            augmented_sentences.append(aug_sentence)
            label_list.append(new_labels)
      
    
    # check for errors
    error_count = 0  
    true_aspects = get_aspect_words(sentence,labels)
    for i in range(len(augmented_sentences)):
        if get_aspect_words(augmented_sentences[i],label_list[i])!=true_aspects:
            error_count+=1
    if error_count>0:
        print(f"detected and removed {error_count} augmented samples")
            
            
        
    # shuffle and trim to num_aug number of sentences
    indices = list(range(len(augmented_sentences)))
    shuffle(indices)
    
    return [(augmented_sentences[idx],label_list[idx]) for idx in indices[:num_aug] if get_aspect_words(augmented_sentences[idx],label_list[idx])==true_aspects]+[(sentence,labels)]


