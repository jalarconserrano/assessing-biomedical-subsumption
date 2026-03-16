from sklearn.metrics import confusion_matrix,f1_score
import traceback
from tqdm import tqdm

import pandas as pd
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.datasets import DataLoader
#from dspy.functional import TypedPredictor
from dspy.primitives.prediction import Prediction
#from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as trace_api
import pydantic
import requests


from lib.ag_assertive import AssertiveClassificationAgent
import pandas

class EvaluatorMy: 

    classlabels=['true','false']
    datacasetypelabels=[
        'TruePositive_Direct', 
        'TruePositive_Transitive', 
        'TrueNegative_Random', 
        'TrueNegative_Inverted', 
        'TrueNegative_InvertedTransitive'
        ]
    
    def __init__(self): 
        self.sampledata = []
        self.datacasetype = [0] * len(EvaluatorMy.datacasetypelabels)
        self.fails = 0
        self.fallstoclassify = 0
        self.assertiveclassify=AssertiveClassificationAgent()        
  
    def test_accuracy(self, example, pred, trace=None):        
        ## check the minimum quality of the prediction
        gold_answer = str(example.answer).lower()
        pred_answer = str(pred.answer).lower()
        
        if not(pred_answer in EvaluatorMy.classlabels):
            if not pred.answer or pred.answer == 'unknown':
                # empty string - discard present test data
                pred_answer='false' if gold_answer=='true' else 'true'
                print(f"Fail: {example}")
                self.fails += 1
            else:                
                pred_answer=str(self.assertiveclassify.classify(pred.answer)).lower()
                print(f"\nClassify: {example,pred}")
                print(f"\nClassify answer: {pred_answer}")
                self.fallstoclassify += 1

        ## save and evaluate prediction
        
        self.sampledata.append( {
            "datacasetype": example.datacasetype,
            "parentdesc": example.parentdesc,
            "childdesc": example. childdesc,
            "answer":  gold_answer,
            "pred": pred_answer,
            "reasoning": (str(pred.reasoning))
        } )
        evaluateresult=(gold_answer == pred_answer)
        self.datacasetype[example.datacasetype]+=1 if not evaluateresult else 0
        return evaluateresult

    def confusion_matrix(self):
        df=pd.DataFrame(self.sampledata)
        return confusion_matrix(df['answer'],df['pred'], labels=EvaluatorMy.classlabels)

    def f1_score(self):
        df=pd.DataFrame(self.sampledata)
        return f1_score(df['answer'],df['pred'],average='weighted')

class AgentTaxRelationSignature(dspy.Signature):
    #context=dspy.InputField(desc="may contain relevant facts")
    childdesc: str=dspy.InputField()
    parentdesc: str=dspy.InputField()
   
    def __init__(self, task):
        super().__init__()
        __doc__ = task
        
class AgentTaxRelationSignatureVanilla(AgentTaxRelationSignature):    
    answer: str=dspy.OutputField(desc='true or false' )

class AgentTaxRelationSignatureTyped(AgentTaxRelationSignature):
    answer: bool=dspy.OutputField(desc='true or false' )

def is_assertive_sentence(pred):
    return (pred.lower()=='true' or pred.lower()=='false')

class AgentTaxRelationModule(dspy.Module):
        
    def __init__(self, task=None, num_passages=0):
        super().__init__()
        if task:
            AgentTaxRelationSignatureVanilla.__doc__=task        
        self.generate_answer = dspy.ChainOfThought(AgentTaxRelationSignatureVanilla)        
  
    def forward(self,parentdesc,childdesc):
        current_span = trace_api.get_current_span()
        auxname=dspy.settings.config['lm'].model
        current_span.set_attribute(SpanAttributes.METADATA, "{ 'model': '"+auxname+"'}")
        try:
            prediction = self.generate_answer(parentdesc=parentdesc,childdesc=childdesc)
        except ValueError:            
            print(f"Error with inputfields: /// {childdesc} /// {parentdesc}")
            traceback.format_exc()
            prediction = Prediction(reasoning='',answer='unknown')
        return prediction

class AgentTaxRelationModuleTyped(dspy.Module):
    def __init__(self, task, num_passages=0):
        super().__init__()
        AgentTaxRelationSignatureTyped.__doc__=task
        self.generate_answer = dspy.TypedChainOfThought(AgentTaxRelationSignatureTyped)
    
    def forward(self,parentdesc,childdesc):
        current_span = trace_api.get_current_span()
        auxname=dspy.settings.config['lm'].model
        current_span.set_attribute(SpanAttributes.METADATA, "{ 'model': '"+auxname+"'}")        
        try: 
            prediction = self.generate_answer(parentdesc=parentdesc,childdesc=childdesc)
        except ValueError:
            print(f"Error with inputfields: /// {childdesc} /// {parentdesc}")
            traceback.format_exc()
            prediction = Prediction(reasoning='',answer='unknown')            
        return prediction


failed_assertion_message = """
Output answer must be a bool value 'true' or 'false' 
Please remove any justification or other text. 
"""

class AgentTaxRelationModuleWithAssertion(dspy.Module):
    def __init__(self, task, num_passages=0):
        super().__init__()
        AgentTaxRelationSignatureVanilla.__doc__=task        
        self.generate_answer = dspy.ChainOfThought(AgentTaxRelationSignatureVanilla)        
    
    def forward(self,parentdesc,childdesc):
        prediction = self.generate_answer(parentdesc=parentdesc,childdesc=childdesc)
        dspy.Suggest(
            is_assertive_sentence(prediction.answer),
            failed_assertion_message
        )
        return prediction

from dspy.teleprompt import MIPROv2

class LearningRelationsAgent:

    def __init__(self,llmargs,optimizer,task,train,test):
        self.llm_name=llmargs['model']
        self.optimizer=optimizer[0]
        self.optimizer_config=optimizer[1]        
        self.task=task
        self.train=train
        self.test=test        
        
        self.predictor_r_ = AgentTaxRelationModule(task=task)
        
        self.ollama_llm=dspy.LM(**llmargs)
        dspy.configure(lm=self.ollama_llm)

    def compile(self):
        evaluatormy = EvaluatorMy()
        teleprompter = self.optimizer(metric=evaluatormy.test_accuracy, **self.optimizer_config)
        self.optimized_program = teleprompter.compile(self.predictor_r_, trainset=self.train,requires_permission_to_run=False)

    def batch_pred(self,dataset):        
        pred_results=[]
        for question_data in tqdm(dataset):
            result_aux=self.predictor_r_(childdesc=question_data.childdesc,parentdesc=question_data.parentdesc)
        
            pred_results.append({
                'childdesc': question_data.childdesc,
                'parentdesc': question_data.parentdesc,
                'rationale': ( result_aux.rationale if hasattr(result_aux,'rationale') else result_aux.reasoning),
                'answer': result_aux.answer
            })

        return pandas.DataFrame(pred_results)
            
    def load(self,path):
        self.predictor_r_ = AgentTaxRelationModule()
        self.predictor_r_.load(path=path)        
    
    def save(self,path):
        if self.optimized_program:
            self.optimized_program.save(path, save_field_meta=True)
            return True
        return False
        
        
    def evaluate(self):
        evaluatormy = EvaluatorMy()
        obj_evaluate = Evaluate(devset=self.test,provide_traceback=True,display_progress=True,display_table=4)

        obj_evaluate(self.predictor_r_, metric=evaluatormy.test_accuracy)
       
        return \
            evaluatormy.confusion_matrix(),\
            evaluatormy.datacasetype,\
            evaluatormy.f1_score(),\
            evaluatormy.fails,\
            evaluatormy.fallstoclassify,\
            pd.DataFrame(evaluatormy.sampledata)
            
    def close_agent(self):
        selected_ip = "127.0.0.1"
        selected_model = self.llm_name.split('/')[1]
        
        url = f"http://{selected_ip}:11434/api/generate"
        
        # Define the payload (data to be sent)
        payload = {
            "model": selected_model,
            "keep_alive": 0
        }
        
        response = requests.post(url, json=payload)
        print(f"Response Status Code: {response.status_code} Response Body: {response.text}")
        
        
        
        
        
        
        