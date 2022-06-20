import torch
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
from utils import *
from datasets import Dataset
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


"""
This script trains a SATSA Model

Input location: "data" folder
Output results location: "logs" folder
"""

class Model(torch.nn.Module):
    """
    model_checkpoint - checkpoints from huggingface
    feature_type[mean,cls,first] - refer to the aspect embeddings
    num_classes - there are 3 classes POS NEG NEU
    """
    def __init__(self, model_checkpoint = "distilbert-base-uncased", feature_type = "mean", num_classes = 3, device = "cuda"):
        
        super(Model, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_checkpoint).to(device)                                        
        self.dropout = torch.nn.Dropout(0.1) 
        feature_size = self.bert.config.hidden_size
        if feature_type in ["cls_last","cls_first","cls_mean"]:                  # concat of cls and aspect embedding would double the size
            feature_size*=2
        self.classifier = torch.nn.Linear(feature_size, num_classes).to(device)   
        self.feature_type = feature_type   
        self.num_span_detection_failed = 0
        self.failed = set()
    """
    given input ids and aspect spans, pass through bert and a classifier head to perform ATSA
    returns the logits for POS, NEG, NEU for each aspect
    """
    def forward(self, input_ids, attention_mask, offsets, aspect_spans, feature_type, device):
        
        contextualized_embeddings = self.bert(input_ids, attention_mask).last_hidden_state  # get bert logits
        contextualized_embeddings = self.dropout(contextualized_embeddings)   
        
        x = self._get_features(embeddings = contextualized_embeddings,                      # get the aspect embeddings/logits/features
                               offsets = offsets, 
                               aspect_spans = aspect_spans, 
                               feature_type = feature_type,
                               device = device)

        logits = self.classifier(x)
        return logits
    """
    given the output embeddings of bert 
    return the embeddings to be used as features 
    options: cls, mean, first
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
def preprocess(dataset, test_size, batch_size, seed = 0):
    dataset["aspect_word"] = dataset.apply(lambda sample: sample["text"][sample["span"][0]:sample["span"][1]], axis=1)
    huggingface_datasets = Dataset.from_pandas(dataset).train_test_split(test_size=test_size, seed = seed)
    huggingface_datasets = huggingface_datasets.map(_tokenize, batch_size = batch_size, batched =True)
    return huggingface_datasets
"""
tokenizes sentences
"""    
def _tokenize(examples, 
              truncation = True,                 
              padding = "longest",            # padding to longest of each batch
              return_offsets_mapping = True): # for aspect embedding extraction 
    if not sentence_b:
        tokenized_examples = tokenizer(examples['text'], 
                                       truncation = truncation, 
                                       padding = padding, 
                                       return_offsets_mapping = return_offsets_mapping)
    else:
        
        tokenized_examples = tokenizer(examples['text'], examples['aspect_word'],
                                       truncation = truncation, 
                                       padding = padding, 
                                       return_offsets_mapping = return_offsets_mapping)
    return tokenized_examples



def evaluate(args, model, eval_dataset, epoch_num):
    
    criterion = torch.nn.CrossEntropyLoss()
    test_size = eval_dataset.num_rows
    running_loss = 0
    num_batches = 0
    y_pred_full = torch.tensor([], dtype = torch.int64).to(args.device)

    for count in tqdm(range(0,test_size,args.batch_size)):

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
                                feature_type = args.feature_type,
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
    save_model_log(args, epoch_num, epoch_loss, epoch_f1, True, "Validation")
    

    
"""

"""
def train(args, model, huggingface_datasets):   
    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.classifier_lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_size = huggingface_datasets['train'].num_rows
    
    
    for epoch in range(1,args.num_epochs+1):

        running_loss = 0
        num_batches = 0
        y_pred_full = torch.tensor([], dtype = torch.int64).to(args.device)
        
        # iterate through training set in mini-batches
        for count in tqdm(range(0,train_size,args.batch_size)):

            optimizer.zero_grad()                                   
            
            # create mini-batch
            minibatch_input_ids =  torch.tensor(huggingface_datasets['train'][count:count+args.batch_size]['input_ids'], 
                                                device = args.device)
            minibatch_attention_mask =  torch.tensor(huggingface_datasets['train'][count:count+args.batch_size]['attention_mask'], 
                                                     device = args.device)
            minibatch_offset_mapping =  huggingface_datasets['train'][count:count+args.batch_size]['offset_mapping']
            minibatch_span =  huggingface_datasets['train'][count:count+args.batch_size]['span']
            minibatch_label =  torch.tensor(huggingface_datasets['train'][count:count+args.batch_size]['label'], device = args.device)
            
            # forward and backward propagation
            logits = model.forward(input_ids = minibatch_input_ids, 
                                    attention_mask = minibatch_attention_mask, 
                                    offsets = minibatch_offset_mapping, 
                                    aspect_spans = minibatch_span, 
                                    feature_type = args.feature_type,
                                    device = args.device) 
            loss = criterion(logits, minibatch_label)
            loss.backward()    # compute gradients 
            optimizer.step()   #update weights

            # this block is for metric computations
            running_loss += loss.detach().item()                # to compute epoch loss
            y_pred_batch = torch.argmax(logits, dim = 1)        # to compute f1
            y_pred_full = torch.cat([y_pred_full,y_pred_batch])
            num_batches+=1
            
        
        # compute epoch loss and error rates 
        epoch_loss = running_loss/num_batches
        epoch_f1 = f1_score(huggingface_datasets['train']['label'], y_pred_full.cpu(), average = "macro")
        save_model_log(args, epoch, epoch_loss, epoch_f1, True, "Train")
        
        # evaluate on validation set
        evaluate(args, model, huggingface_datasets['test'], epoch)
        
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

            
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--feature_type', type=str, default="mean", help='Options: [mean, first, last, cls]')
    parser.add_argument('--encoder_name', type=str, default="distilbert-base-uncased", help='Any HuggingFace Transformer Encoder')
    parser.add_argument('--trainable_parameters', type=list, default=["classifier.weight","classifier.bias"], help='to train encoder layers, specify its names')
    parser.add_argument('--classifier_lr', type=float, default=0.01, help='classifier learning rate')
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--dataset_name', type=str, default="MAMS", help = "Options:[MAMS, Fi_ATSA, FiQA]")
    parser.add_argument('--dataset_frac', type=float, default=1.0, help = "to make dataset smaller for test runs")
    parser.add_argument('--random_state', type=int, default=0, help = "this shuffles dataset")
    parser.add_argument('--sentence_b', type=bool, default=False, help = "whether to use aspect as sentence b")
    parser.add_argument('--train_mode', type=bool, default=True, help = "if train mode, load train set, otherwise dev set")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    # seed
    set_random_seed(args.random_state)
    
    # model components
    print("Instantiating model components..")
    global tokenizer, sentence_b
    sentence_b = args.sentence_b
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = Model(num_classes = 3, 
                  model_checkpoint = args.encoder_name, 
                  feature_type = args.feature_type, 
                  device = args.device)
    set_trainable_parameters(model, trainable_parameters = args.trainable_parameters)
    print("Model and tokenizer instantiated!")
    
    # dataset
    print("Loading and processing dataset..")
    pandas_dataset = load_dataset(args.dataset_name, 
                                  random_state = args.random_state, 
                                  dataset_frac = args.dataset_frac)
    huggingface_datasets = preprocess(pandas_dataset, 
                                      test_size = args.test_size, 
                                      batch_size = args.batch_size,
                                      seed = args.random_state)
    print("Dataset load and processed!")
    
    # run train
    print("Training..")
    train(args, model, huggingface_datasets)
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