from transformers import pipeline
import itertools
import torch

class AssertiveClassificationAgent:

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #TODO: jdas01 RECUPERAR ESTO
        self.sentiment_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=self.device)
        self.candidate_labels = ['afirmative', 'negative', 'unknown']

    def classify(self,sentence):
        #TODO: jdas01 RECUPERAR ESTO
        #return True
        predictions=self.sentiment_classifier(sentence,self.candidate_labels)
        for prediction,score in zip(predictions['labels'],predictions['scores']):
            if ( score == max(predictions['scores']) ):
                if prediction == 'afirmative':
                    return True
                elif prediction == 'negative':
                    return False
                else: 
                    return None