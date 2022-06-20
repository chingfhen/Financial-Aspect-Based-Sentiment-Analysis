import os
import csv


PATHS = {
    "MAMS" : './data/MAMS/train/mams_atsa_train.pkl',
    "Fi_ATSA": './data/Fi_ATSA/train/Fi_ATSA.pkl',
    "FiQA":'./data/FiQA/train/FiQA_train.pkl',
    "Fi_ATSA_train": './data/Fi_ATSA/train/Fi_ATSA_train.pkl',        # 70% 
    "Fi_ATSA_eda": './data/Fi_ATSA/train/Fi_ATSA_eda.pkl',            # eda on 70% set
    "Fi_ATSA_test": './data/Fi_ATSA/train/Fi_ATSA_test.pkl',          # 30%
}
DEV_PATHS = {
    "MAMS" : './data/MAMS/dev/mams_atsa_train.pkl',
    "Fi_ATSA_test": './data/Fi_ATSA/train/Fi_ATSA_test.pkl',          # 30%
    "FiQA":None
}

import pandas as pd
from random import randint
from IPython.display import display, HTML
def fully_show_samples(dataset, num_samples = 10, randomize = True):
    if isinstance(dataset,pd.DataFrame):                  # if DataFrame 
        if randomize:                                          # if random> shuffle
            dataset = dataset.sample(frac=1)
        display(HTML(dataset.iloc[:num_samples].to_html()))             # take first n rows
    else:                                                    # if not DataFrame
        if randomize:                                           # if random> shuffle
            dataset = dataset.shuffle(seed = randint(0,100))   
        dataset = pd.DataFrame(dataset.select(range(num_samples)))   # convert first n rows to dataframe
        display(HTML(dataset.to_html()))
        
        

def set_trainable_parameters(model, trainable_parameters):
    for name, param in model.named_parameters():
        if name in trainable_parameters:
            param.requires_grad = True
            print(f"Trainable: {name}")
        else: 
            param.requires_grad = False
            
def reset_trainable_parameters(model, trainable_parameters):
    for i, (layer_name, layer) in enumerate(model.named_children()):
        if layer_name in trainable_parameters:
            layer.reset_parameters()
    print("Classifier resetted!")
            
            
def load_dataset(dataset_name, random_state = 0, dataset_frac = 1.0):
    path = PATHS[dataset_name]
    dataset = pd.read_pickle(path).sample(frac=dataset_frac, random_state = random_state).reset_index(drop = True)
    return dataset


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
    
    
def save_model_log(args, epoch, epoch_loss, epoch_f1, show_results = True, mode = "Train"):
        
    encoder_name = args.encoder_name.replace("/","-")
    filename = f"{args.dataset_name}-{encoder_name}-{args.feature_type}-{args.batch_size}-{args.random_state}-{args.sentence_b}" 
    PATH = './logs/text_files/'+ filename + ".txt"
    if not os.path.isfile(PATH):
        with open(PATH, 'a') as f:
            description = f'Dataset: {args.dataset_name} \nEncoder_name: {encoder_name} \nFeature_type: {args.feature_type}\nBatch_size: {args.batch_size}\n'
            f.write(description)
    with open(PATH, 'a') as f:
        results = f'{mode} mode: Epoch {epoch}, loss {epoch_loss}, f1 {epoch_f1}\n'
        f.write(results)
    if show_results:
        print(results)
        
def save_model_log_2(args, epoch, epoch_loss, epoch_f1, num_batches = None, mode = "Train"):
    encoder_name = args.encoder_name.replace("/","-")
    filename = f"{args.dataset_name}-{encoder_name}-{args.HL1_size}-{args.learning_rate}-{args.batch_size}-{args.random_state}" 
    PATH = './logs/csv_files/'+ filename + ".csv"
    if not os.path.isfile(PATH):
        with open(PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            header = [
                "dataset_name", "encoder_name", "HL1_size", "learning_rate", "batch_size", "random_state",
                "epoch","epoch_loss", "epoch_f1","mode", "num_batches"
            ]
            writer.writerow(header)
    with open(PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [
            args.dataset_name, encoder_name, args.HL1_size, args.learning_rate, args.batch_size, args.random_state,
            epoch, epoch_loss, epoch_f1, mode , num_batches
        ]
        writer.writerow(row)
        
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
"""
FOR VISUALISATIONS
"""

"""
Get all results from folder
folder: str - contains text files with the results
output: pandas dataframe
"""
def get_all_results(folder):
    results_df = pd.DataFrame()
    for filename in os.listdir(folder):
        results = pd.DataFrame(_extract_results(filename,folder))  
        results_df = results_df.append(results)
    results_df.f1 = results_df.f1.astype("float")
    results_df.loss = results_df.loss.astype("float")
    results_df = results_df[results_df.loss<=30]
    results_df['feature_type2'] = results_df.apply(lambda row: _process(row) ,axis=1)
    results_df.epoch = results_df.epoch.astype(int)
    return results_df

"""
Extract results from text file 
filename: str - contains results
folder: str - folder containing filename
output: list of dictionaries
"""
def _extract_results(filename, folder):
    container = []
    path =os.path.join(folder,filename)
    results0 = filename.split(".")[0].split("-")
    data_name, feature_type, batch_size, seed, two_sentence = results0[0], results0[-4], results0[-3], results0[-2], results0[-1]
    with open(path, "r") as f:
        for l in f.readlines():
            results1 = l.split(": ")
            results1[-1] = results1[-1].replace("\n","").strip()
            if "Encoder_name"==results1[0]:
                encoder_name = results1[-1]
            elif "mode" in results1[0]:
                epoch, loss, f1 = results1[-1].split(", ")
                epoch, loss, f1 = map(lambda x: x.split()[-1], [epoch, loss, f1])
                mode = "validation" if "Validation" in results1[0] else "train"
                container.append({
                    "data_name":data_name, 
                    "feature_type":feature_type, 
                    "batch_size":batch_size, 
                    "seed":seed, 
                    "two_sentence":two_sentence,
                    "encoder_name":encoder_name,
                    "epoch":epoch, 
                    "loss":loss, 
                    "f1":f1,
                    "mode":mode
                })
    return container

def _process(row):
    if row['feature_type']=="cls":
        return "cls_two_sentence"
    elif row['two_sentence']=="True":
        return f"{row['feature_type']}_two_sentence"
    return row['feature_type']

        
        
    


            
            
