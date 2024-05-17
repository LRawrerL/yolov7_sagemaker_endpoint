# Why host AI on the cloud?
By hosting a computer vision model on the cloud like Yolo (You only look one) we are able to gain these benefits:
- Scalability
- Cost reduction
- Utilise powerful computational resources
- Less management

One of these cloud services that offer these benefits is AWS SageMaker

![](https://i.imgur.com/oPycvZ7.png)
[Unsplash](https://unsplash.com/photos/white-clouds-K-Iog-Bqf8E)

# What is SageMaker?
SageMaker is a service that helps automate and simplify much of the pipeline from training to deployment. It is a Platform as a Service (PaaS) which helps manage many of the underlying infrastructure of hosting and training AI on the cloud. 

![](https://i.imgur.com/5a8VrVH.png)

## What is a SageMaker Endpoint
Endpoints are a key feature for deployment. They allow you to publish your trained models to the cloud and make them accessible. 

There is a few inferencing options, we will be using the Real-Time Inference as real time object detection we will need low latency and high throughput -> [More Options Here](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)

## How does a SageMaker Endpoint work 
You might wish to know how SageMaker endpoints operate inside before creating any code.

![](https://i.imgur.com/EYv83VB.png)

This example uses fastapi as our backend; however you can use any type of backend. SageMaker will deploy your docker container with all the artefacts (weights etc) and inference code.  

# Pre-req
- AWS Account 
- A configured IAM role with SageMakerFullAccess and PriceListServiceFullAccess
- AWS CLI installed and configured
- Trained AI Model
- Docker Environment Set up
- Using Linux/Mac Environment
 
>[!note] 
>Dont worry, as I will be going through on how to configure the IAM Role and AWS CLI Installation

# Why Docker
We are using a more customised model that SageMaker does not offer the PyTorch that YoloV7 requires. (As the time of writing) Also it helps simplify our deployment as we can easily control the environment and packages to be installed. Hence, the algorithm implementation we will be using is Custom Image.

![](https://i.imgur.com/YgSgbAA.png)

# Step 1 Creating a IAM User and role
Now we will need to create a IAM User through AWS
- Firstly in the search bar in AWS type **IAM**
- Then on the left find **User**
- Afterwards create a **User** with FullAccess to AWS as a policy attaached
- Then generate a **Access Key** and **Secret Access Key**, keep this saved somewhere for now

Do the same with a role; However only include FullAccess to SageMaker and FullAccess to PriceListService as policies. Then save the ARN 

![](https://i.imgur.com/sjNWvtF.png)

# Step 2 Install and Configuring AWS CLI
We will be following the official AWS Guide here -> [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on how to install the command

*For Linux*
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

After verifying the installation and the being able to run the `aws` command we will now need to used the saved keys to configure our AWS CLI

*Universal*
```bash
aws configure
AWS Access Key ID []: <Your Key>
AWS Secret Access Key []: <Your Key>
Default region name []: <The region you want to deploy the endpoint>
Default output format []: <json recommended>
```

Now your code will be able to access AWS Services and informations.

# Step 3 Creating the Backend

## Folder Structure
This is my current folder structure, you can have it organised the way you want.

```
├── models  
|-- utils 
├── Dockerfile  
├── inference_backend.py
|-- entry.py
|-- weights.pt
└── requirements.txt
```
- models and utils are folders from the YoloV7 github
- inference_client.py is where I load my yolo model and inference_helper.py and inference_frame.py are scripts that will run the detection and results
- entry.py is my uvicorn wrapper

## Requirements

```txt
# The requirements that YoloV7 needs and any extras that you will be using

# Endpoint requirements
fastapi
uvicorn[standard]
python-multipart
```

## Backend API
In order to deploy your backend server we need to handle two specific end points that [Adapting Own Inference Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html) specifies by AWS
- `POST` to `/invocations`
- `POST` to `/ping`

Afterwards you need to make sure your backend uses port `8080`

I am using [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) as a wrapper around it. However you can chose to use Flask etc.

*inference_endpoint.py* 
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
"""
Import anything else that is a depedency for loading the model etc
"""

"""
Parameters for command line
User parser and load them all them
"""

app = FastAPI()

# Enable CORS for all routes (you may want to restrict this in a production environment)
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# Load pre-trained ResNet50 model
model = < Your way of loading the model >

# Define transformation for the input image
"""
Transform your image using transforms if needed
"""

@app.post("/invocations")
async def predict(request: Dict[Any, Any]):
	"""
	Code that handles the image meta data that has
	been sent to the endpoint
	"""
	return JSONResponse(content={"""Json Content to return"""}, status_code=200)

@app.get("/ping")
async def ping():
	"""
	However you want to respond to the /ping, can be
	, think of it as a health check to check to see if 
	everything is loaded correctly
	"""
	return JSONResponse(content={"""Json Content to return"""}, status_code=200)
```

>[!note] The code on top is a template 
>Just know you need a way of **loading your model** through your **pt** file and also to **process results and return them**

*entry.py*
```python
from inference_backend import app
import uvicorn

if __name__ == "__main__":
	import uvicorn
	uvicorn.run("inference_backend:app", host="0.0.0.0", port=8080, reload=False, workers=1)
```

## Docker File
```D
# Set the base image
FROM --platform=linux/amd64 python:3.8-slim-buster

# Set the working directory
ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program

# Copy the requirements file
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
gcc \
python3-dev \
libgl1-mesa-dev \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender-dev

# Install the requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8080 for the app
EXPOSE 8080

# Run the application
ENTRYPOINT ["python", "entry.py", "--name"]
```

Standard Docker file, installs all of the dependencies and ensures we are building for the architecture it will be deployed on `--platform=linux/amd64` 

The reason why we use `--name` at the `ENTRYPOINT` command is when SageMaker runs the container it does `Docker run <image name> serve` this will cause a extra argument to be passed into the python script. Hence it will become `python entry.py --name serve`. This will help reduce errors when running the command

# Step 4 Uploading To ECR
We will be continuing to use this guide here ->  [Adapting Own Inference Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html) 

I have made a few changes to the **Bash Script** below to fit my needs.
*Bash Script*
```bash
#!/bin/bash
algorithm_name=SageMakerModel
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
region=${region:-us-east-1}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
	sudo aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}
sudo docker build -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
```

Afterwards you should see your container in **AWS ECR**
*Name may not be the same*
![](https://i.imgur.com/8fjFKAZ.png)

# Step 5 Launching the Endpoint
I have compiled all the steps from -> [Adapting Own Inference Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html) into one python file that will assume the IAM role instead of using pass down.

It will then also save the endpoint name into a txt file which can you use to read later.

```python
import boto3
from sagemaker import get_execution_role
# Create an STS client
sts_client = boto3.client('sts')
role = <Insert ARN>
# Assume the role
assumed_role_object = sts_client.assume_role(
RoleArn= role,
RoleSessionName="AssumeRoleSession1"
)
# From the response that's returned by the API, extract the temporary
# credentials that can be used to make subsequent API calls
credentials = assumed_role_object['Credentials']
# Use the temporary credentials to create a new session
session = boto3.Session(
aws_access_key_id=credentials['AccessKeyId'],
aws_secret_access_key=credentials['SecretAccessKey'],
aws_session_token=credentials['SessionToken'],
)
# Now you can use this session to create clients for other services
sm_client = session.client('sagemaker')
runtime_sm_client = session.client('sagemaker-runtime')
region = session.region_name
account_id = boto3.client('sts').get_caller_identity()['Account']

from time import gmtime, strftime
model_name = 'Yolov7-model' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

container = '{}.dkr.ecr.{}.amazonaws.com/SageMakerModel:latest'.format(account_id, region)

instance_type = 'ml.m5.xlarge' #cpu
print('Model name: ' + model_name)
#print('Model data Url: ' + model_url)
print('Container image: ' + container)
container = {
'Image': container
}

create_model_response = sm_client.create_model(
ModelName = model_name,
ExecutionRoleArn = role,
Containers = [container])

print("Model Arn: " + create_model_response['ModelArn'])

endpoint_config_name = 'Yolov7-config' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print('Endpoint config name: ' + endpoint_config_name)

create_endpoint_config_response = sm_client.create_endpoint_config(
EndpointConfigName = endpoint_config_name,
ProductionVariants=[{
'InstanceType': instance_type,
'InitialInstanceCount': 1,
'InitialVariantWeight': 1,
'ModelName': model_name,
'VariantName': 'AllTraffic'}])
print("Endpoint config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

import time
endpoint_name = 'Yolov7-endpoint' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print('Endpoint name: ' + endpoint_name)
create_endpoint_response = sm_client.create_endpoint(
EndpointName=endpoint_name,
EndpointConfigName=endpoint_config_name)
print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp['EndpointStatus']
print("Endpoint Status: " + status)

print('Waiting for {} endpoint to be in service...'.format(endpoint_name))
waiter = sm_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)

with open('endpoint_name.txt', 'w') as file:
	file.write(endpoint_name)
```

# Step 6 Invoking Endpoint

This is just a template client code to show you image processing through invoking the template

```python
import boto3

def send_image():
	with open('endpoint_name.txt', 'r') as file:
		for line in file:
			endpoint_name = line.strip()
		file.close()
	
	runtime_sm_client = boto3.client(service_name
	='sagemaker-runtime')
	
	"""
	Convert image to metadata which will be send in 
	json format
	"""
	
	content_type = "application/json"
	request_body = {"input": """Meta Data"""}
	payload = json.dumps(request_body)
	
	#Endpoint invocation
	response = runtime_sm_client.invoke_endpoint(
	EndpointName=endpoint_name,
	ContentType=content_type,
	Body=payload)
	
	#Parse results
	"""
	Code that can make use of the inference results
	"""
```

After invocation of the end point and parsing the inference results we are able to do object detection using YoloV7 over the cloud

![](https://i.imgur.com/RzhLbA3.png)
