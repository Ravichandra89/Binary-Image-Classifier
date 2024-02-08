import boto3

region = "ap-south-1"
bucket_name = 'imageclassifieanimal'
model_key = 'models/animal_injury_classifier.h5'  # Update with the correct path to your model file in S3
model_name = "animal-injury-model"
instance_type = 'ml.t2.medium'  

# Create a SageMaker client
sagemaker = boto3.client('sagemaker', region_name=region)

# Step 1: Specify the location of the trained model in S3
model_url = f's3://{bucket_name}/{model_key}'

# Step 2: Create a SageMaker model
sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': '763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04',
        'ModelDataUrl': model_url,
    },
    ExecutionRoleArn='arn:aws:iam::770065223990:role/Role'
)

# Step 3: Create an endpoint configuration
endpoint_config_name = 'animal-injury-endpoint-config'
response = sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': instance_type
        }
    ]
)

# Step 4: Create an endpoint
endpoint_name = 'animal-injury-endpoint'
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Wait for the endpoint to be in service
waiter = sagemaker.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)

print("Endpoint created successfully!")
