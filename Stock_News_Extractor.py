import pandas as pd
import lxml
import re
import requests
from random import sample
import spacy
import os
import argparse
from tqdm import tqdm
import jsonlines


"""
This script makes eodhistoricaldata financial news API calls to generate Aspect Term Sentiment Analysis annotation samples:

Output format: .jsonl files
{"sentence": "Fils-Aime pointed to successful acquisitions ...", "spans": [[50, 70], [121, 130], [154, 173]], "span_values": ["Microsoft", "Activision Blizzard", "Take-Two Interactive"]}
{"sentence": "Despite challenges ahead, some investors see Microsoft's ...", "spans": [[45, 54], [58, 64], [97, 103]], "span_values": ["Microsoft", "NASDAQ"]}

output location:
"tmp" folder

Note: These samples are NOT sentiment annotated, but they can be loaded into annotation tools like Doccano: https://github.com/doccano/doccano

Arguments: 
--api_key ; your eodhistoricaldata api key (required)
--min_aspects ; minimum number of aspects per sentence, aspects are all SP500 company names like "Microsoft" and "Apple" 
--specific_company_ticker ; ticker of the company to call from eodhistoricaldata news api
--unique_aspects_only ; whether each sample must have unique aspects

Example usage: 
python MAMS_Stock_News_Extractor.py --api_key 621ed1bbc56a96.1088xxxx --specific_company_ticker INTC --min_aspects 2 --unique_aspects_only True
"""


"""
Given a json output from eodhistoricaldata, produce MAMS samples. 
Samples should contain at least min_aspects.
"""
def MAMS_sample_generate(eodhistoricaldata_json, aspect_list, min_aspects = 2):
    nlp = spacy.load('en_core_web_sm')
    output = {"sentence":[],"spans":[],"date":[]}
    aspects_regex = _get_regex_to_match(aspect_list)
    document = _extract_text(eodhistoricaldata_json)
    sentence_generator = nlp(cleanText(document)).sents
    for sentence in sentence_generator:
        span_object = _get_span_indices(sentence.text, aspects_regex)
        if len(span_object)>=min_aspects:
            output["sentence"].append(sentence.text)
            output["spans"].append([span.span() for span in span_object])
            output['date'].append(eodhistoricaldata_json["date"])
    return output

"""
Makes 1 call from eodhistoricaldata for ticker
"""
def call_eodhistoricaldata_api(ticker, api_key):
    url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={ticker}'
    response = requests.get(url)
    response.raise_for_status()
    eodhistoricaldata_json = response.json()
    return eodhistoricaldata_json 

# returns regex that matches company names ... \\bAlbemarle\\b|\\bAlexandria Real Estate Equities\\b ...
def _get_regex_to_match(strings_to_match: list):
    for i,string in enumerate(strings_to_match):
        strings_to_match[i] = r"\b"+re.sub("\([^()]*\)","",string).strip()+r"\b"
    return "|".join(strings_to_match)

# Extracts start and end position of aspect/company name
def _get_span_indices(text:str, regex):
    return [i for i in re.finditer(regex,text,re.IGNORECASE)]

# concats the title and content of a eodhistorical json data
def _extract_text(json):
    return json['title']+". "+json['content']

# purpose: removes urls, excess spaces and newlines etc.
# input: str
# output: str
def cleanText(text):                                       
    def remove_links(text):
        return re.sub(r'http\S+', '', text)                    
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_escape_sequence_char(text):
        return text.replace('\\', '').replace(u'\xa0', u'').replace('\n', '').replace('\t', '').replace('\r', '')
    return white_space_fix(remove_escape_sequence_char(remove_links(text)))
    
# used in main() to extract samples with multiple unique spans    
def get_span_values(sentence, spans, unique_only):
    if unique_only:
        return list({sentence[s[0]:s[1]] for s in spans})
    return [sentence[s[0]:s[1]] for s in spans]




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required = True, help='eodhistoricaldata api key')
    parser.add_argument('--min_aspects', type=int, default=2, help='minimum number of aspects per sentence')
    parser.add_argument('--num_calls', type=int, default=1, help='number of calls from eodhistoricaldata')
    parser.add_argument('--specific_company_ticker', default=None, help='if not specified a random sp500 company is chosen')
    parser.add_argument('--unique_aspects_only', type = bool ,default=True, help='sentences must have more than min_aspects aspects that are unique (or not)')
    args = parser.parse_args()
    
    # get aspect list; a list of sp500 comapany names 
    print("Getting SP500 Company names and symbols..")
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    SP500 = payload[0]
    aspect_list = SP500.Security.to_list()
    symbol_list = SP500.Symbol.to_list()
    symbol_list.remove("T")
    aspect_list.remove("Nasdaq")
    aspect_list.remove("Target")
    print("Done!")
    
    
    for i in range(1,args.num_calls+1):
        
        # final output container
        container = pd.DataFrame()
        
        # company ticker to call from eodhistoricaldata
        if args.specific_company_ticker:
            ticker = args.specific_company_ticker
        else:
            ticker = sample(symbol_list,1)[0]    # sample 1 company from sp500 companies
        
        print(f"Call number {i}:")
        print(f"Ticker: {ticker}")
        # call from eodhistoricaldata
        json_list = call_eodhistoricaldata_api(ticker, args.api_key)
        print("Call Success!")

        # generate MAMS samples
        print("Generating Samples")
        for j in tqdm(json_list):
            container = container.append(pd.DataFrame(MAMS_sample_generate(j, aspect_list,args.min_aspects)))
            
        # extract samples with multiple unique aspects
        container["span_values"] = container.apply(lambda row: get_span_values(row["sentence"],row["spans"], args.unique_aspects_only), axis=1)
        if args.unique_aspects_only:
            container = container[container['span_values'].apply(lambda x: len(x)>=args.min_aspects)]
        print(f"{len(container)} samples generated!")

        # save samples to tmp folder as pickle 
        print(f"Saving {ticker} samples..")   
        filename = f"{ticker}_MAMS_Stock_News_Samples.jsonl" 
        current_directory = os.getcwd()
        tmp_folder = os.path.join(current_directory, "tmp")
        if not os.path.isdir(tmp_folder):
            os.mkdir(tmp_folder)
        save_path = os.path.join(tmp_folder, filename)
        with jsonlines.open(save_path, 'w') as writer:
            writer.write_all(container.to_dict('records'))
        print("Done!")
    
    
if __name__ == '__main__':
    main()