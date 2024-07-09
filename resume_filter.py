import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging
import numpy as np
import pickle
from transformers import pipeline
import sqlite3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeFilter:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.embeddings = None

    def load_data(self, file_path):
        try:
            self.df = pd.read_excel(file_path)
            self.df['text'] = self.df.apply(self.combine_columns, axis=1)
            logger.info("Data loaded and combined successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def combine_columns(row):
        return ' '.join([str(row['candidateName']), str(row['companyName']), str(row['designation']),
                         str(row['experienceMas']), str(row['qualificationMas']), str(row['qualificationMas2'])])

    def get_bert_embedding(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating BERT embedding: {e}")
            raise

    def generate_embeddings(self, batch_size=32, save_path='embeddings.pkl'):
        try:
            embeddings = []
            for i in tqdm(range(0, len(self.df), batch_size), desc="Generating embeddings"):
                batch_texts = self.df['text'][i:i+batch_size].tolist()
                batch_embeddings = [self.get_bert_embedding(text) for text in batch_texts]
                embeddings.extend(batch_embeddings)
            self.df['embedding'] = embeddings
            self.embeddings = np.array(embeddings)
            with open(save_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info("Embeddings generated and saved successfully.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def load_embeddings(self, load_path='embeddings.pkl'):
        try:
            with open(load_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info("Embeddings loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def filter_resumes(self, job_description, top_n=5):
        try:
            job_embedding = self.get_bert_embedding(job_description)
            similarities = cosine_similarity(self.embeddings, [job_embedding]).flatten()
            self.df['similarity'] = similarities
            filtered_df = self.df.sort_values(by='similarity', ascending=False).head(top_n)
            return filtered_df[['candidateName', 'companyName', 'designation', 'similarity']]
        except Exception as e:
            logger.error(f"Error filtering resumes: {e}")
            raise


if __name__ == "__main__":
    file_path = '/content/drive/MyDrive/TN EMPLOYEE DATABASE.xlsx'  
    resume_filter = ResumeFilter()
    resume_filter.load_data(file_path)
    resume_filter.generate_embeddings()

    while True:
        job_description = input("Enter the job description (or type 'exit' to quit): ").strip()
        if job_description.lower() == 'exit':
            print("Exiting the program.")
            break
        filtered_resumes = resume_filter.filter_resumes(job_description)
        print(filtered_resumes)



extractor = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
text = "Your job offer or resume text here"
extracted_info = extractor(text)
print(extracted_info)

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


conn = sqlite3.connect('embeddings.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, embedding BLOB)''')

