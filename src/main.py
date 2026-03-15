import os
import json

import time
from datetime import datetime

from lib.ag_relations import LearningRelationsAgent,EvaluatorMy
from lib.reportsutils import getsimplifiedmodelname, radar_chart,confusionmatrix_chart
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import traceback

from dspy.datasets import DataLoader
import numpy as np

import dspy
import phoenix as px
from phoenix.client import Client

from openinference.semconv.resource import ResourceAttributes
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter \
    import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

import sys

with open("config.json") as json_data_file:
    static_configs = json.load(json_data_file)


def wait_for_traces(client, expected_min, project_name, timeout=240, poll=2):
    start = time.time()
    last_count = -1

    while True:
        try: 
            traces = client.get_trace_dataset(project_name,
                                              limit=-1,
                                              timeout=None
                                              )           
            count = len(traces)
            if count >= expected_min and count == last_count:
                return count

        except:
            count=0

        
        if time.time() - start > timeout:
            return 0

        last_count = count
        time.sleep(poll)

class OutputMan(object):
    def __init__(self, filename):
        self.file = open(filename, "a")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def isatty(self):
        return self.stdout.isatty()

    def fileno(self):
        return self.stdout.fileno()

    def close(self):
        self.file.close()

    def __getattr__(self, name):
        return getattr(self.stdout, name)

timestampstr=datetime.now().strftime('%Y%m%d%H%M')
sys.stdout = OutputMan(f"main_output_{timestampstr}.log")
sys.stderr = sys.stdout

# Prep.
current_directory = os.getcwd()
print("Working directory is:", current_directory)
DATADIR="../data"
DATADIRSOURCE="../data/prepared"
DATADIROUTPUT="../data/results"

## Observability - Phoenix Setup
timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
resource = Resource(attributes={
    ResourceAttributes.PROJECT_NAME: f"llmsknowledge_{timestamp}"
   })
phoenix_session = px.launch_app()
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(span_exporter=span_otlp_exporter)    
    )
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument(skip_dep_check=True)
print(phoenix_session.url)


# 1st. Load data set
df=pd.read_csv(
    f"{DATADIRSOURCE}/{static_configs['datasetname']}",
    delimiter='|',
    dtype={'answer': str}
    )
dl = DataLoader()
snomed_relations_dataset = dl.from_pandas(
    df,
    input_keys=['childdesc','parentdesc'],
    fields={'datacasetype','childdesc','parentdesc','answer'}
    )

splits = dl.train_test_split( # `dataset` must be a List of dspy.Example
    snomed_relations_dataset, 
    test_size=0.2
    )

train_dataset = splits['train']
test_dataset = snomed_relations_dataset # all data

df_test=pd.DataFrame(
    [(
      line_dataset.datacasetype,
      line_dataset.parentdesc,
      line_dataset.childdesc,
      line_dataset.answer
      )
     for line_dataset in test_dataset
     ],columns=['datacasetype','parentdesc', 'childdesc', 'answer']
    )
df_train=pd.DataFrame(
    [(
      line_dataset.datacasetype,
      line_dataset.parentdesc,
      line_dataset.childdesc,
      line_dataset.answer
      )
     for line_dataset in train_dataset
     ],columns=['datacasetype','parentdesc', 'childdesc', 'answer']
    )

print (f"## TIMESTAMP: {timestampstr} ##")


test_metadata=pd.Series(
    [ex.answer for ex in test_dataset]).value_counts().to_dict()
print(f"""
\nTest: 
\n{test_metadata}
\n
""")

print(test_dataset[1])

train_metadata=pd.Series(
    [ex.answer for ex in train_dataset]).value_counts().to_dict()
print(f"""
\nTrain: 
\n{train_metadata}
\n""")

print(train_dataset[1])


# 2nd. define tasks and models 
workflow=static_configs['workflow']
MODEL_NAMES=static_configs['MODEL_NAMES']
TASKS=static_configs['TASKS']
OPTIMIZERS=static_configs['OPTIMIZERS']

## - main loop - ## 


grouped_df = df_test.groupby(df_test.columns[0]).count()
experimentdata={
    't': len(df_test),
    't_true': test_metadata['true'] if 'true' in test_metadata else 0,
    't_false': test_metadata['false'] if 'false' in test_metadata else 0,
    'cases_t_by_type': [0] * len(EvaluatorMy.datacasetypelabels),
    'models': list(),
    }

for aux in grouped_df.index: 
    experimentdata['cases_t_by_type'][aux]=grouped_df.parentdesc[aux]

    
for modelcfg,task,optimizer in itertools.product(MODEL_NAMES,TASKS,OPTIMIZERS):
    if isinstance(optimizer[0], str):
        optimizer[0]=eval(optimizer[0])
    n_model=getsimplifiedmodelname(modelcfg['model'])
    stampfiles=str(time.time() * 1000).split('.')[0]
    print (f"## STAMP_filenames: {stampfiles} ##")
    print (f"""
## {n_model} 
##### task: {task}
##### optimizer: {optimizer[0].__name__}
""")
    
    test_case=LearningRelationsAgent (
        llmargs=modelcfg,
        optimizer=optimizer,        
        task=task,
        train=train_dataset,
        test=test_dataset,        
    )
    experimentdata['models'].append(f"{n_model}_{stampfiles}")
    
    # 3rd. run agents and collect results for each model
    directory=f"{DATADIROUTPUT}/{modelcfg['model'].split(':')[0]}_{stampfiles}"
    os.makedirs(directory, exist_ok=True)
    
    start_t = time.perf_counter()
    if workflow and workflow=='compile':
        try:
            test_case.compile()
            end_t = time.perf_counter()
            elapsed = end_t - start_t
            print(f'Time taken: {elapsed:.6f} seconds')
            test_case.save(f"{directory}.{optimizer[0].__name__}_{stampfiles}.json")
        except ValueError:        
            print(traceback.format_exc())
    else: 
        optimize_prompt_file=f"{directory}.{optimizer[0].__name__}_{stampfiles}.json"
        is_optimize_prompt="zeroshot"
        if os.path.exists(optimize_prompt_file):            
            test_case.load(optimize_prompt_file)
            is_optimize_prompt={optimizer[0].__name__}
            print(f'Loaded optimize agent from: {optimize_prompt_file}')
            
        confusionmatrix,datacasetypes,f1,fails,fallstoclassify,sampledata=test_case.evaluate()
        end_t = time.perf_counter()
        experimentdata[f"{n_model}_{stampfiles}"]={
            'confusionmatrix': confusionmatrix,
            'task': task,
            'fails_t_by_type' : datacasetypes,
            'fails_avg_by_type': [
                a / b if a>0 else 0 
                for a,b in zip(datacasetypes,experimentdata['cases_t_by_type'])
                ],
            'f1': f1,
            'fails': fails,
            'fallstoclassify': fallstoclassify,
            'performance_evaluate': ( end_t - start_t ) / experimentdata['t'], #time per request TODO
        }        
        
        sampledata.to_csv(f'{DATADIROUTPUT}/test_{n_model}.{stampfiles}.csv', sep='|')
        print(f"Fails by category in {n_model}: {datacasetypes}")
                
        #4th. Present results
        confusionmatrix_chart(n_model,confusionmatrix)
        plt.savefig(f"{directory}.{is_optimize_prompt}_{stampfiles}.svg")
        plt.show()
    
    
    project_name=f"llmsknowledge_{timestamp}"
    wait_for_traces(px.Client(), project_name=project_name, expected_min=5)
    pxds=px.Client().get_trace_dataset(
        project_name=project_name,
        limit=-1,
        timeout=None
        )
    pxds.get_spans_dataframe().to_parquet(path=f"{DATADIROUTPUT}/{n_model}_{timestampstr}.parquet")
    

print (experimentdata) # show a table with f1-score, performance and faults
np.save(f'{DATADIROUTPUT}/results_sumary.{timestampstr}.exp',experimentdata)
# to load: experimentdata = np.load(f'{DATADIROUTPUT}/results_sumary.{timestampstr}.exp.npy',allow_pickle=True).item()

df_test.to_csv(f'{DATADIROUTPUT}/test_dataset.{timestampstr}.csv', sep='|')
df_train.to_csv(f'{DATADIROUTPUT}/train_dataset.{timestampstr}.csv', sep='|')

dmodels=dict()
for aux in experimentdata['models']:
    dmodels[aux]=experimentdata[aux]['fails_avg_by_type']
    
radar_chart(
    labels=EvaluatorMy.datacasetypelabels, 
    namemodels=experimentdata['models'],
    dataframe=dmodels,
    title=is_optimize_prompt
    )
plt.savefig(f'{DATADIROUTPUT}/radar_chart.{timestampstr}.svg')
plt.show()

dataframe_experiment=pd.DataFrame(
    data=[
            [aux, #model name
            1/experimentdata[aux]['performance_evaluate'], # predictions / seg.
            1-sum(experimentdata[aux]['fails_t_by_type'])/experimentdata['t'], # accuracy (fails / total)
            experimentdata[aux]['f1']] # f1 - score 
            for aux in experimentdata['models']
        ],
    columns=['model','performance_evaluate','accuracy','f1']
    )
    
print(dataframe_experiment)






