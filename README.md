# scones_project
1. All the code  is in the file scones_notebook_complete_local.ipynb and it can be viewed as html as well.
I modified the code to be able to run in my local docker container that was built from public AWS sagemaker image. I was training in SageMaker in AWS, but I was using aws client to access, deploy, run services from my docker container

2. I did POC of lambda / step function through regular python functions. I reimplemnted them as AWS step and lambda functions and attached results as required

3. I downloaded lambda individually as zip files as I could not find a value in merging all of them into the same file as each represents an independent service. The lambda functions code attached below.

4. I spent a lot of time to create sagemaker lambda layer as the requirements were requesting python 3.8 environment, libraries that were coming with sagemaker installation were not compatible with this version, I had to manually truncate list of packages in the layer. Unfortunately this work took most allocated time otherwise I would spent on more productive work related to machine learning


```

# Process input
import json

import base64
import os
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the S3 bucket and key from the Step Function event input
    key = event['s3_key']  # Assuming the event has 's3_key'
    bucket = event['s3_bucket']  # Assuming the event has 's3_bucket'
    
    # Download the data from s3 to /tmp/image.png
    download_path = '/tmp/image.png'
    s3.download_file(bucket, key, download_path)
    
    # We read the data from the file and encode it
    with open(download_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')  # Convert bytes to string
    
    # Log the event details (for debugging purposes)
    print("Event:", event)
    
    # Return the response to the Step Function
    return {
        'statusCode': 200,
        'body': json.dumps({
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []  # Empty, to be filled by further steps in Step Function
        })
    }
    
# Inference call

import json
import sagemaker
import base64
import boto3
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor


ENDPOINT = 'image-classification-2024-09-26-21-52-25-323'

def lambda_handler(event, context):
    """A function to run inference on an image using SageMaker"""

    image = base64.b64decode(event['body']['image_data'])

    predictor = Predictor(endpoint_name=ENDPOINT)   
    predictor.serializer = IdentitySerializer("image/png")
    inferences = predictor.predict(image)
    event["inferences"] = inferences.decode('utf-8') 
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

# Classify the result 
import json

THRESHOLD = .93


def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event['inferences']
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (inferences[0] > THRESHOLD)
    print(inferences[0])

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode' : 200,
        'body' : json.dumps(event)
    }

```



