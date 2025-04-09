import requests
import json
import time
from fuzzywuzzy import fuzz
import gc
import pandas as pd



class crossref_matcher:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)


    def make_harvard_citation(self, retrieved_data):
        """
        Generates a Harvard-style citation from the crossref retrieved data.
        """
        citation = ''
            
        if retrieved_data.get('author'):
            for author in retrieved_data['author']:
                if author.get('given'): 
                    citation += f"{author['given']} "
                if author.get('family'):
                    citation += f"{author['family']}, "
        
        if retrieved_data.get('issued'):
            citation += f"{retrieved_data['issued']['date-parts'][0][0]}. "

        if retrieved_data.get('title'):
            citation += f"{retrieved_data['title'][0]}. "
            
        if retrieved_data.get('short-container-title'):
            citation += f"{retrieved_data['short-container-title'][0]}. "
        
        if retrieved_data.get('volume'):
            citation += f"{retrieved_data['volume']}, "
        
        if retrieved_data.get('page'):
            citation += f"pp. {retrieved_data['page']}."

        return citation
    
    def process_parsed_data(self):
        """
        Fetches the matches from the CrossRef API and compares them with the formatted references.
        """
        for _, row in self.df.iterrows():
            ref = row['formatted_reference']
            ref_url = requests.utils.quote(ref)
            
            url = f'http://api.crossref.org/works/?query.bibliographic={ref_url}'
            response = requests.get(url)
            data = json.loads(response.text)
            
            retrieved_citation = self.make_harvard_citation(data['message']['items'][0])
            ratio_score = fuzz.ratio(ref, retrieved_citation)
            
            if ratio_score < 50:
                ## LLM to determine whether it is a match
                print(ref)
                print(retrieved_citation)
                print(ratio_score)
                print('-'*len(ref))
            
            time.sleep(0.2)
            gc.collect()
