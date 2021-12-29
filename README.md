# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classification data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
The pretrained model of choice for this project is Resnet18. Resnet is one of the best-performing pretrained datasets used for image recognition. 

For the hyperparameter tuning, I included the following hyperparameters and ranges:

1. learning rate (lr) with a range of .001 - .1
2. batch size with options 16, 32, 64, 128
3. epochs with a range of 2 to 5

These hyperparameters were chosen because they have the greatest effect on a model's performance.

Hyperparameter tuning jobs successfully completed
![](hyperparameter-tuning.png)

Log metrics from hyperparameter tuning job
![](hpt-logs.jpg)

Best parameters
![](best-parameters.jpg)

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
