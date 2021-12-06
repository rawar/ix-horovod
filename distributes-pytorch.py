#!/usr/bin/env python
# coding: utf-8

# # Distributed PyTorch with Horovod

# In[40]:


# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)


# In[2]:


from azureml.telemetry import set_diagnostics_collection

set_diagnostics_collection(send_diagnostics=True)


# In[3]:


import uuid

# Create a model folder in the current directory
os.makedirs('./model', exist_ok=True)
timeline_dir = "./model/horovod-timeline/%s" % uuid.uuid4()
os.makedirs(timeline_dir, exist_ok=True)
os.environ['HOROVOD_TIMELINE'] = timeline_dir + "/horovod_timeline.json"


# In[4]:


from azureml.core.workspace import Workspace

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')


# ## Attach existing Cluster

# In[24]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "ramwar-gpu-node"
compute_target = ComputeTarget(workspace=ws, name=cluster_name)
print('Found existing cluster.')
print(compute_target.get_status().serialize())


# ## Train model on the remote compute

# In[32]:


import os

project_folder = './pytorch-distr-hvd'
os.makedirs(project_folder, exist_ok=True)


# ### Create an experiment

# In[33]:


from azureml.core import Experiment

experiment_name = 'pytorch-distr-hvd'
experiment = Experiment(ws, name=experiment_name)


# ### Create an environment

# In[34]:


from azureml.core import Environment

pytorch_env = Environment.get(ws, name='AzureML-PyTorch-1.6-GPU')


# ### Configure the training job

# In[41]:


from azureml.core import ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

src = ScriptRunConfig(source_directory=project_folder,
                      script='pytorch_horovod_mnist.py',
                      #script='pytorch_synthetic_benchmark.py',
                      compute_target=compute_target,
                      environment=pytorch_env,
                      distributed_job_config=MpiConfiguration(node_count=3))


# ### Submit Job
# 
# As the run is executed, it goes through the following stages:
# 
# * __Preparing__: A docker image is created according to the environment defined. The image is uploaded to the workspace's container registry and cached for later runs. Logs are also streamed to the run history and can be viewed to monitor progress. If a curated environment is specified instead, the cached image backing that curated environment will be used.
# 
# * __Scaling__: The cluster attempts to scale up if the Batch AI cluster requires more nodes to execute the run than are currently available.
# 
# * __Running__: All scripts in the script folder are uploaded to the compute target, data stores are mounted or copied, and the script is executed. Outputs from stdout and the ./logs folder are streamed to the run history and can be used to monitor the run.
# 
# * __Post-Processing__: The ./outputs folder of the run is copied over to the run history.
# 

# In[42]:


run = experiment.submit(src)
print(run)


# ### Register model

# In[37]:


model = run.register_model(model_name='pytorch-mnist', model_path='outputs/model.pt')


# In[43]:


from azureml.widgets import RunDetails

RunDetails(run).show()


# ### Waiting for training completion

# In[44]:


run.wait_for_completion(show_output=True) # this provides a verbose log


# ### Download model

# In[20]:


# Download the model from run history
run.download_file(name='outputs/model.pt', output_file_path='./model/model.pt')
#run.download_file(name='outputs/model.onnx', output_file_path='./model/model.onnx')

