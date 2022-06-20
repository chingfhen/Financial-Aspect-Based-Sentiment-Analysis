import torch
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
from utils import *
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


"""
This script trains a SATSA Model using FinBERT and a neural classifier taking cls+aspect embeddings as input.
This script also allows for hyperparameter tuning of neural classifier
"""

"""
Input location: "data" folder
Output results location: "logs" folder
"""
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

class Model(torch.nn.Module):
    """
    cross validation to fairly compare parameters
    can be trained on any valid dataset (eda or no eda)
    must have early stopping
    evaluation of best model on test set
    """
    def __init__(self, HL1_size = 10, device = "cuda"):
        
        super(Model, self).__init__()
        """
        ProsusAI/finbert found to be best 
        cls_last feature_type - taking cls + last aspect embedding found to be best
        classifier - 2 layers
        """
        self.model_checkpoint = "ProsusAI/finbert"     
        self.feature_type = "cls_last"
        self.bert = AutoModel.from_pretrained(self.model_checkpoint).to(device)                                        
        self.dropout = torch.nn.Dropout(0.1)
        self.HL1 = torch.nn.Linear(self.bert.config.hidden_size*2, HL1_size).to(device) 
        self.BN1 = torch.nn.BatchNorm1d(HL1_size).to(device) 
        self.classifier = torch.nn.Linear(HL1_size, 3).to(device)             
        
        self.num_span_detection_failed = 0
        self.failed = set()
        
    """
    given input ids and aspect spans, pass through bert and a classifier head to perform ATSA
    returns the logits for POS, NEG, NEU for each aspect
    """
    def forward(self, input_ids, attention_mask, offsets, aspect_spans, feature_type, device):
        
        contextualized_embeddings = self.bert(input_ids, attention_mask).last_hidden_state  # get bert logits
        contextualized_embeddings = self.dropout(contextualized_embeddings)   
        x = self._get_features(embeddings = contextualized_embeddings,                      # get the embeddings as input features
                               offsets = offsets, 
                               aspect_spans = aspect_spans, 
                               feature_type = feature_type,
                               device = device)
        x = self.HL1(x)            
        x = self.BN1(x)               
        x = F.relu(x)

        logits = self.classifier(x)
        return logits
    """
    given the output embeddings of bert 
    return the embeddings to be used as features 
    options: "cls_last","cls_first","cls_mean"
    """
    def _get_features(self, embeddings, offsets, aspect_spans, feature_type, device):   # mean, first, cls
        if feature_type=="cls":  
            # cls is the embedding at index 0
            features = torch.tensor(np.array([embedding[0].cpu().detach().numpy() for embedding in embeddings], dtype = "float32")).to(device)                    
        elif feature_type in ["last","first","mean"]:
            # either takes the first,last or mean aspect embedding 
            features = np.array([embedding[self._is_aspect_span(offsets[i],aspect_spans[i])].cpu().detach().numpy() for i,embedding in enumerate(embeddings)], dtype=object)
            if feature_type=="first":
                features = torch.tensor(np.array([ft[0] for ft in features], dtype = "float32")).to(device)
            elif feature_type=="last":
                features = torch.tensor(np.array([ft[-1] for ft in features], dtype = "float32")).to(device)
            elif feature_type=="mean":
                features = torch.tensor(np.array([np.mean(ft,axis=0) for ft in features], dtype = "float32")).to(device)
            
        # refers to concatenation of cls embedding and first/last/mean aspect embedding
        elif feature_type in ["cls_last","cls_first","cls_mean"]:
            cls_features = torch.tensor(np.array([embedding[0].cpu().detach().numpy() for embedding in embeddings], dtype = "float32")).to(device)  
            aspect_features = np.array([embedding[self._is_aspect_span(offsets[i],aspect_spans[i])].cpu().detach().numpy() for i,embedding in enumerate(embeddings)], dtype=object)
            if feature_type=="cls_first":
                aspect_features = torch.tensor(np.array([ft[0] for ft in aspect_features], dtype = "float32")).to(device)
            elif feature_type=="cls_last":
                aspect_features = torch.tensor(np.array([ft[-1] for ft in aspect_features], dtype = "float32")).to(device)
            elif feature_type=="cls_mean":
                aspect_features = torch.tensor(np.array([np.mean(ft,axis=0) for ft in aspect_features], dtype = "float32")).to(device)
            features = torch.concat([cls_features,aspect_features],axis=1)
        return features    
    

    """
    given the offsets [[0,0],[0,5],...,] and aspect span (0, 10) 
    return an array indicating which are the tokens are aspects [F,T,T,F,F..]
    if the aspect was not found, let cls be the aspect (there likely is an error with the span input)
    """
    def _is_aspect_span(self, offsets, aspect_span, must_have = True):
        boolean_array = np.zeros(len(offsets), dtype = bool)
        found = False
        for i,OS in enumerate(offsets):
            if OS[1]==0:
                continue
            elif OS[0]>=aspect_span[0] and OS[1]<=aspect_span[1]:
                boolean_array[i]=True
                found = True
            elif found:
                return boolean_array
        if must_have:
            self.num_span_detection_failed += 1  
            self.failed.add(tuple(aspect_span))
            boolean_array[0] = True
        return boolean_array
        
"""   
given pandas dataset, does processing and returns huggingface Dataset
leverage batch processing of huggingface
"""
def preprocess(dataset, batch_size, seed = 0):
    dataset["aspect_word"] = dataset.apply(lambda sample: sample["text"][sample["span"][0]:sample["span"][1]], axis=1)
    huggingface_datasets = Dataset.from_pandas(dataset)
    huggingface_datasets = huggingface_datasets.map(_tokenize, batch_size = batch_size, batched =True)
    return huggingface_datasets
"""
tokenizes sentences
"""    
def _tokenize(examples, 
              truncation = True,                 
              padding = "max_length",            
              max_length = 100,                # padding to 100 
              return_offsets_mapping = True): # for aspect embedding extraction 
    tokenized_examples = tokenizer(examples['text'], 
                                   truncation = truncation, 
                                   padding = padding, 
                                   max_length = max_length,
                                   return_offsets_mapping = return_offsets_mapping)
    return tokenized_examples



def evaluate(args, model, eval_dataset, epoch_num, num_batches):
    
    criterion = torch.nn.CrossEntropyLoss()
    test_size = eval_dataset.num_rows
    running_loss = 0
    num_batches = 0
    y_pred_full = torch.tensor([], dtype = torch.int64).to(args.device)

    for count in range(0,test_size,args.batch_size):

        # create mini-batch
        minibatch_input_ids =  torch.tensor(eval_dataset[count:count+args.batch_size]['input_ids'], device = args.device)
        minibatch_attention_mask =  torch.tensor(eval_dataset[count:count+args.batch_size]['attention_mask'], device = args.device)
        minibatch_offset_mapping =  eval_dataset[count:count+args.batch_size]['offset_mapping']
        minibatch_span =  eval_dataset[count:count+args.batch_size]['span']
        minibatch_label =  torch.tensor(eval_dataset[count:count+args.batch_size]['label'], device = args.device)

        # forward and backward propagation
        logits = model.forward(input_ids = minibatch_input_ids, 
                                attention_mask = minibatch_attention_mask, 
                                offsets = minibatch_offset_mapping, 
                                aspect_spans = minibatch_span, 
                                feature_type = model.feature_type,
                                device = args.device) 

        # this block is for metric computations
        loss = criterion(logits, minibatch_label)
        running_loss += loss.detach().item()                # to compute epoch loss
        y_pred_batch = torch.argmax(logits, dim = 1)        # to compute f1
        y_pred_full = torch.cat([y_pred_full,y_pred_batch])
        num_batches+=1

    # compute epoch loss and error rates 
    epoch_loss = running_loss/num_batches
    epoch_f1 = f1_score(eval_dataset['label'], y_pred_full.cpu(), average = "macro")

    
    save_model_log_2(args, epoch_num, epoch_loss, epoch_f1, 
                     num_batches = num_batches, mode = "Validation")
    

    
    
    
def evaluate_on_test(args, model, dataset_name = "Fi_ATSA_test", nb = None):
    
    print("Loading and processing test dataset..")
    pandas_dataset = load_dataset(dataset_name,
                                  dataset_frac = args.dataset_frac)
    huggingface_dataset = preprocess(pandas_dataset, 
                                     batch_size = args.batch_size)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_size = huggingface_dataset.num_rows
    running_loss = 0
    num_batches = 0
    y_pred_full = torch.tensor([], dtype = torch.int64).to(args.device)

    print("Evaluating on test dataset..")
    for count in range(0,test_size,args.batch_size):
        # create mini-batch
        minibatch_input_ids =  torch.tensor(huggingface_dataset[count:count+args.batch_size]['input_ids'], device = args.device)
        minibatch_attention_mask =  torch.tensor(huggingface_dataset[count:count+args.batch_size]['attention_mask'], device = args.device)
        minibatch_offset_mapping =  huggingface_dataset[count:count+args.batch_size]['offset_mapping']
        minibatch_span =  huggingface_dataset[count:count+args.batch_size]['span']
        minibatch_label =  torch.tensor(huggingface_dataset[count:count+args.batch_size]['label'], device = args.device)

        # forward and backward propagation
        logits = model.forward(input_ids = minibatch_input_ids, 
                                attention_mask = minibatch_attention_mask, 
                                offsets = minibatch_offset_mapping, 
                                aspect_spans = minibatch_span, 
                                feature_type = model.feature_type,
                                device = args.device) 

        # this block is for metric computations
        loss = criterion(logits, minibatch_label)
        running_loss += loss.detach().item()                # to compute epoch loss
        y_pred_batch = torch.argmax(logits, dim = 1)        # to compute f1
        y_pred_full = torch.cat([y_pred_full,y_pred_batch])
        num_batches+=1
    
    print("Saving result..")
    # compute epoch loss and error rates 
    epoch_loss = running_loss/num_batches
    epoch_f1 = f1_score(huggingface_dataset['label'], y_pred_full.cpu(), average = "macro")
    save_model_log_2(args, "NA", epoch_loss, epoch_f1, mode = "Test", num_batches = nb)
    

    
"""

"""
def train(args, model):  
    
    
    # dataset
    print("Loading and processing dataset..")
    pandas_dataset = load_dataset(args.dataset_name, 
                                  random_state = args.random_state,
                                  dataset_frac = args.dataset_frac)
    huggingface_datasets = preprocess(pandas_dataset, 
                                      batch_size = args.batch_size,
                                      seed = args.random_state)
    print("Dataset load and processed!")
    
    
    KF = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)
    
    for i,(train_indices,valid_indices) in enumerate(KF.split(range(huggingface_datasets.num_rows))):
        
        reset_trainable_parameters(model, args.trainable_parameters)
        
        print(f"Training on fold {i}..")
        train_set, validation_set = huggingface_datasets.select(train_indices),huggingface_datasets.select(valid_indices)
    
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        train_size = train_set.num_rows
        
        num_batches_before_validation = int(train_size/args.batch_size/args.num_validation_per_epoch) if args.num_validation_per_epoch else float("inf")
        
        for epoch in tqdm(range(1,args.num_epochs+1)):

            running_loss = 0
            num_batches = 0
            y_pred_full = torch.tensor([], dtype = torch.int64).to(args.device)

            # iterate through training set in mini-batches
            for count in range(0,train_size,args.batch_size):

                optimizer.zero_grad()                                   

                # create mini-batch
                minibatch_input_ids =  torch.tensor(train_set[count:count+args.batch_size]['input_ids'], 
                                                    device = args.device)
                minibatch_attention_mask =  torch.tensor(train_set[count:count+args.batch_size]['attention_mask'], 
                                                         device = args.device)
                minibatch_offset_mapping =  train_set[count:count+args.batch_size]['offset_mapping']
                minibatch_span =  train_set[count:count+args.batch_size]['span']
                minibatch_label =  torch.tensor(train_set[count:count+args.batch_size]['label'], device = args.device)

                # forward and backward propagation
                logits = model.forward(input_ids = minibatch_input_ids, 
                                        attention_mask = minibatch_attention_mask, 
                                        offsets = minibatch_offset_mapping, 
                                        aspect_spans = minibatch_span, 
                                        feature_type = model.feature_type,
                                        device = args.device) 
                loss = criterion(logits, minibatch_label)
                loss.backward()    # compute gradients 
                optimizer.step()   #update weights

                # this block is for metric computations
                running_loss += loss.detach().item()                # to compute epoch loss
                y_pred_batch = torch.argmax(logits, dim = 1)        # to compute f1
                y_pred_full = torch.cat([y_pred_full,y_pred_batch])
                num_batches+=1
                
                
                if num_batches%num_batches_before_validation==0:
                    evaluate_on_test(args, model, dataset_name = args.test_dataset_name, nb = num_batches)
#                     evaluate(args, model, validation_set, epoch, num_batches = num_batches)

            # compute epoch loss and error rates 
#             epoch_loss = running_loss/num_batches
#             epoch_f1 = f1_score(train_set['label'], y_pred_full.cpu(), average = "macro")
#             save_model_log_2(args, epoch, epoch_loss, epoch_f1, num_batches = num_batches, mode = "Train")
#             evaluate_on_test(args, model, dataset_name = args.test_dataset_name)
            # evaluate on validation set at the end of each epoch
#             if args.num_validation_per_epoch==0:
#                 evaluate(args, model, validation_set, epoch, num_batches = None)
    
        # evaluate on test set at the end of each training fold
        if args.test_dataset_name is not None:
            pass
#             evaluate_on_test(args, model, dataset_name = args.test_dataset_name)
            
    
        
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

            
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1 , type = int)
    parser.add_argument('--dataset_name', default = "Fi_ATSA_train", type = str,
                        help = "Options:[Fi_ATSA, Fi_ATSA_train, Fi_ATSA_eda, Fi_ATSA_test]")
    parser.add_argument('--HL1_size', default=10, type = int,
                        help='number of hidden neurons')
    parser.add_argument('--learning_rate', default=0.001, type = float,
                        help='classifier learning rate')
    parser.add_argument('--batch_size', type = int,
                        default=16)
    parser.add_argument('--random_state', default=0, type =int ,  
                        help = "seed for numpy, torch and dataset")
    
    parser.add_argument('--num_validation_per_epoch', default=0, type =int ,
                        help = "Number of times to validate every epoch")
    
    parser.add_argument('--trainable_parameters',  type = list, 
                        default=["HL1.weight", "HL1.bias", "BN1.weight", "BN1.bias", "classifier.weight", "classifier.bias", 
                                 "HL1", "BN1", "classifier"], 
                        help='to train encoder layers, specify its names')
    parser.add_argument('--num_folds', default=3, type = int)
    parser.add_argument('--dataset_frac', default=1.0, type = float)
    parser.add_argument('--test_dataset_name', default=None, type = str,
                        help = "Options: [Fi_ATSA_test]")
    parser.add_argument('--encoder_name', type=str, default="ProsusAI/finbert", help='Options: [ProsusAI/finbert]')
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    # seed
    set_random_seed(args.random_state)
    
    # model components
    print("Instantiating model components..")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = Model(HL1_size = args.HL1_size,
                  device = args.device)
    set_trainable_parameters(model, trainable_parameters = args.trainable_parameters)
    print("Model and tokenizer instantiated!")
    
    
    # run train
    print("Training..")
    train(args, model)
    print("Training Complete!")
    
    print(model.failed)            # these are spans that are likely annotated wrongly  

if __name__ == '__main__':
    main()
    

    
#    def _get_features(self, embeddings, offsets, aspect_spans, feature_type, device):   # mean, first, cls
#         if feature_type=="cls":  
#             # cls is the embedding at index 0
#             features = torch.tensor(np.array([embedding[0].cpu().detach().numpy() for embedding in embeddings], dtype = "float32")).to(device)                    
#         else:
#             # either takes the first aspect embedding or the mean
#             features = np.array([embedding[self._is_aspect_span(offsets[i],aspect_spans[i])].cpu().detach().numpy() for i,embedding in enumerate(embeddings)], dtype=object)
#             if feature_type=="first":
#                 features = torch.tensor(np.array([ft[0] for ft in features], dtype = "float32")).to(device)
#             elif feature_type=="last":
#                 features = torch.tensor(np.array([ft[-1] for ft in features], dtype = "float32")).to(device)
#             elif feature_type=="mean":
#                 features = torch.tensor(np.array([np.mean(ft,axis=0) for ft in features], dtype = "float32")).to(device)
                
#             torch.concat([x,x],axis=1)
        
#         return features   