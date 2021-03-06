# tf-keras-gcloud-deployment
Training and deploying a TensorFlow tf.keras model using Google Cloud ML Engine

## Introduction

While trying to deploy a tf.keras model using Google Cloud Platform (GCP) ML Engine, we ran into many challenges and found the documentation lacking in many ways. Once things were finally working, we decided to put it all together in this repo to potentially help others who are doing the same thing. This codebase focuses on the cloud deployment side of things. All of the model training is done locally, but there are many tutorials for how to train on the cloud as well. The deployment scenario also mainly deals with image inputs, though handles them in different ways (including a simple list string, a base64 encoding, and GCP Storage link).

What is shown here is just one way to get this whole process working. The specific configurations may not be optimal for everyone's use case. Additionally, this field moves very quickly and the APIs are often changing so what works now may not work in 6 months.


## Setup

This code is tested in python 3.6 on Mac OSX 10.13 with an Anaconda distribution (though all dependencies are reinstalled). The python libraries required are all installed within a virtual environment. If you don't have virtualenv, install it before proceeding. In the project root directory, run the following:

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```


### Setting up TensorFlow

In this codebase we are using TensorFlow version 1.10 without a GPU. We will install it directly using the following command:

```
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl
```

If not on Mac, follow the TensorFlow installation instructions [here](https://www.tensorflow.org/install/pip).


## Model Training and Preparation

In this example, we are training a very simple fully convolutional model on a toy dataset. The output heatmaps capture the brightest parts of the image. This problem is not very interesting, but it will allow us to train a model quickly and use it for deployment. The multidimensional nature of the output makes the deployment a bit more interesting, but all the steps below should be the same for a flat output such as what you'd have with a simple classification model.

### Model Training

In the root directory of the repo, run:

```
python train.py
```


### Exporting the Model

Here we need to convert our tf.keras .h5 model so the SavedModel format that is compatible with GCP deployment. There are quite a few complexities in setting everything up to correctly work. The biggest challenge is getting the input data handled correctly for the desired format. This StackOverflow [post](https://stackoverflow.com/questions/51432589/how-do-i-get-a-tensorflow-keras-model-that-takes-images-as-input-to-serve-predic) was particularly helpful in getting everything right.

In this code, we will export three different models that are set up to handle data in different formats. When calling "gcloud ml-engine predict", the input is always a JSON file, but the contents within the JSON file vary depending on the desired input type. The following formats are supported:
1. Image converted to a list and written as a string in the JSON file
    - This is the simplest approach, but the least efficient and for larger images, the JSON file exceeds the filesize limit
2. JPEG image bytes written to a string using a base64 encoding
    - More complex, but smaller JSON file size
3. Image is just a URL in the JSON file (generally pointing to an image file stored in some bucket in GCP Storage)

Run the following code to export the three different models. They'll end up in the "models" folder.

```
python export_models.py
```

NOTE: The base64 and url methods assume that the images are always jpeg. The code will need to be adapted to work with other image types. Just switching to decode\_image instead of decode\_jpeg does not solve the problem because decode\_image has variable output shape because of the GIF format.

NOTE: This section doesn't do much to optimize the size of the model. There are other actions that can be taken in the export process to optimize the model for inference. This [tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) has some helpful information.


## General GCP Configuration

To start off, you will need to have a GCP account. If this is your first time creating a GCP account, you may get some free credits. Once you have an account, follow the instructions below to get everything set up for this demo.

Follow steps 1-6 in the "Set up your GCP project" [here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).
- In step 1, first select your organization, then click on "CREATE PROJECT", name the project "tf-keras-deploy"
- Step 2 might not be necessary as billing may be set up by default for the project
- In step 3, go to the linked page and select the "tf-keras-deploy" in the lower dropdown and click "Continue"
    - This takes a couple of minutes to process
    - Eventually the page will reload with a "Go to credentials" link
- Follow the exact directions in step 4
    - Find the downloaded JSON file and move it into ~/.gcloud (make this directory first)
- Steps 5 and 6 are pretty straightforward


## GCP Bucket Setup

To deploy a model to GCP, you need to have a storage bucket on GCP where the model resides. We will also use this bucket for sample data for the deployment option that just passes the URL when calling predict. Below are a quick set of instructions for setting up a bucket from the console. See the first 4 steps in this [tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set_up_your_storage_name_short_bucket) for more details.

NOTE: if you already had the Google cloud SDK installed on your machine, you will probably need to set the active project to the one we created in the above steps. You can do this with the following terminal command:

```
gcloud config set project tf-keras-deploy
```

Be sure that the last parameter above is the project_id, not the project name. In some cases, they may be different from each other (for example if you delete a project and then create a new one with the same name, the id will not match the name).

To create a new bucket run the following commands in the terminal:

```
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
echo $BUCKET_NAME
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
```


## Input Preparation

Before moving on to testing, we will prepare some JSON input files that have the correct data format. Some of the docs suggest a JSON input format that looks something like this:

```
{
  "instances": [
    {"input_key": image_content},
    {"input_key": image_content}       
  ]
}
```

This may be necessary for processing a set of images. For single image processing, we found that it was necessary to use the following simpler format:

```
{"input_key": image_content}
```

In both examples above, the "input\_key" should be consistent with the input key used in [export_models.py](export_models.py). For base64 encodings, this key must end with the string "bytes". The image content will differ depending on the input type being used. For the image as list string, this content is just a big string with the list. For the URL input version, the image_content is just a string with the URL. For the base64 encoded input, the image content is actually another dictionary that looks like this:

```
{"b64": base64_encoded_string}
```

To generate these sample input JSON files for a test image in "data/test", run the following script:

```
python image_to_json.py
```

Note that the base64 encoded JSON file is about 60 times smaller than the file with the image values written out as a list string. This is the main motivation for using the base64 encoding.


## Model Deployment and Testing

In this example, we will create a single model and then use different versions of the model to test the different types of input. In production you might want to break these out into separate models depending on the requirements, but for this demo we went with the simplest path. Instructions for the model deployment process can be found [here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#deploy_a_model_to_support_prediction) or [here](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models#creating_a_model_version). However, you can also just run the following terminal commands:

```
MODEL_NAME=highlights_fcn
gcloud ml-engine models create $MODEL_NAME --regions=$REGION
gcloud ml-engine models list
```

Below, we provide examples for the 3 different input scenarios. In each case, we first upload the appropriate model files to the bucket we created (we'll put them each inside a top level folder called "models"). Then we use the ml-engine versions create tool to create a new model version. Finally, we test the deployed model to make sure we get back reasonable results.


### Local model testing

Before we jump into deploying and testing the models, let's first run our test image through our trained model locally to see what the model predictions should look like.

```
python evaluate.py --image_path data/test/test_img.jpg \
                   --heatmap_path data/test/test_heatmap.jpg \
                   --output_name local
```

This should produce something like the image below though the exact output will differ some because each trained model will vary based on random initializations.

![Image showing local model predictions](figs/preds_local.jpg "Local Predictions")


### Image as list string

Set up some bash variables specific to this input type:

```
EXPORT_VERSION=v1     # this should be whatever is hardcoded in export_models.py
INPUT_TYPE=json_list  # this is also hardcoded in export_models.py
VERSION_NAME=${INPUT_TYPE}_${EXPORT_VERSION}
echo $VERSION_NAME
BINARY_DIR_NAME=$(ls -1 models/$INPUT_TYPE/$EXPORT_VERSION)
LOCAL_BINARIES=models/$INPUT_TYPE/$EXPORT_VERSION/$BINARY_DIR_NAME
echo $LOCAL_BINARIES
REMOTE_BINARIES=gs://$BUCKET_NAME/$LOCAL_BINARIES
```

Upload the model to the cloud and create a model version for this input type:

```
gsutil cp -r $LOCAL_BINARIES $REMOTE_BINARIES
gcloud ml-engine versions create $VERSION_NAME \
                                 --model $MODEL_NAME \
                                 --origin $REMOTE_BINARIES \
                                 --runtime-version 1.10
```

The creation of the model version takes a little while to process. Once it is done, you can verify that the model is there with the following command:

```
gcloud ml-engine versions list --model $MODEL_NAME
```

Now that the model is all set, we can call the model to get a prediction:

```
gcloud ml-engine predict --model $MODEL_NAME \
                         --version $VERSION_NAME \
                         --json-instances data/test/test_json_list.json \
                         > preds/test_json_list.txt
```

Create a visualization of the result to make sure it looks right:

```
python evaluate.py --image_path data/test/test_img.jpg \
                   --heatmap_path data/test/test_heatmap.jpg \
                   --output_name list \
                   --text_preds_path preds/test_json_list.txt
```

Compare this to the local predictions figure to be sure it looks right.


### Image as base64 string

Set up some bash variables specific to this input type:

```
EXPORT_VERSION=v1    # this should be whatever is hardcoded in export_models.py
INPUT_TYPE=json_b64  # this is also hardcoded in export_models.py
VERSION_NAME=${INPUT_TYPE}_${EXPORT_VERSION}
echo $VERSION_NAME
BINARY_DIR_NAME=$(ls -1 models/$INPUT_TYPE/$EXPORT_VERSION)
LOCAL_BINARIES=models/$INPUT_TYPE/$EXPORT_VERSION/$BINARY_DIR_NAME
echo $LOCAL_BINARIES
REMOTE_BINARIES=gs://$BUCKET_NAME/$LOCAL_BINARIES
```

Upload the model to the cloud and create a model version for this input type:

```
gsutil cp -r $LOCAL_BINARIES $REMOTE_BINARIES
gcloud ml-engine versions create $VERSION_NAME \
                                 --model $MODEL_NAME \
                                 --origin $REMOTE_BINARIES \
                                 --runtime-version 1.10
```

The creation of the model version takes a little while to process. Once it is done, you can verify that the model is there with the following command:

```
gcloud ml-engine versions list --model $MODEL_NAME
```

Now that the model is all set, we can call the model to get a prediction:

```
gcloud ml-engine predict --model $MODEL_NAME \
                         --version $VERSION_NAME \
                         --json-instances data/test/test_json_b64.json \
                         > preds/test_json_b64.txt
```

Create a visualization of the result to make sure it looks right:

```
python evaluate.py --image_path data/test/test_img.jpg \
                   --heatmap_path data/test/test_heatmap.jpg \
                   --output_name base64 \
                   --text_preds_path preds/test_json_b64.txt
```

Compare this to the local predictions figure to be sure it looks right.


### Image as URL

To test passing the image as a URL, we first need to upload the test image into the bucket we created for this project. In general, this approach is useful when you already have a bunch of data uploaded to Google storage and you want to run it all through the model. Run the following command to upload the test image into the bucket:

```
gsutil cp data/test/test_img.jpg gs://$BUCKET_NAME/data/test/test_img.jpg 
```

NOTE: the remote file path must match what gets written to the JSON input file in the [image_to_json.py](image_to_json.py) script

Next we repeat a similar set of commands as above.

Set up some bash variables specific to this input type:

```
EXPORT_VERSION=v1    # this should be whatever is hardcoded in export_models.py
INPUT_TYPE=json_url  # this is also hardcoded in export_models.py
VERSION_NAME=${INPUT_TYPE}_${EXPORT_VERSION}
echo $VERSION_NAME
BINARY_DIR_NAME=$(ls -1 models/$INPUT_TYPE/$EXPORT_VERSION)
LOCAL_BINARIES=models/$INPUT_TYPE/$EXPORT_VERSION/$BINARY_DIR_NAME
echo $LOCAL_BINARIES
REMOTE_BINARIES=gs://$BUCKET_NAME/$LOCAL_BINARIES
```

Upload the model to the cloud and create a model version for this input type:

```
gsutil cp -r $LOCAL_BINARIES $REMOTE_BINARIES
gcloud ml-engine versions create $VERSION_NAME \
                                 --model $MODEL_NAME \
                                 --origin $REMOTE_BINARIES \
                                 --runtime-version 1.10
```

The creation of the model version takes a little while to process. Once it is done, you can verify that the model is there with the following command:

```
gcloud ml-engine versions list --model $MODEL_NAME
```

Now that the model is all set, we can call the model to get a prediction:

```
gcloud ml-engine predict --model $MODEL_NAME \
                         --version $VERSION_NAME \
                         --json-instances data/test/test_json_url.json \
                         > preds/test_json_url.txt
```

HACK ALERT:

When the above command is executed, the "preds/test\_json\_url.txt" actually contains an error message saying that the cloud ML service is not able to access the bucket with the data. This a bit strange because the bucket is part of the same project. From the IAM section of the GCP web interface, the cloud ml service appears to have the correct permissions and there is no service name that actually matches the service mentioned in the error message. This problem is similar to what is documented in this StackOverflow [question](https://stackoverflow.com/questions/49054125/cloud-ml-service-account-cannot-access-cloud-storage-and-is-not-listed-in-iam) except concerns the data access and not the model access (the model access already worked for us above). The question is not definitively answered though.

It is possible that this problem is unique to our particular setup, but if you encounter it as well, follow the next steps to resolve it:

1. Find the name of the service that "does not have storage.objects.get access"
    - Should look something like "cloud-ml-service-cml@12345-cml.iam.gserviceaccount.com"
2. Log in to GCP web portal and click on the "Storage" section
3. Be sure to select the "tf-keras-deploy" project at the top
4. Click on the "tf-keras-deploy-mlengine" bucket
5. Select the "Permissions" tab
6. Click "Add members"
7. Copy the name of the service into the text input
8. Select "Storage Object Viewer" for "Select a role" and click "Add"

Once you have done this, rerun the command above and it should work this time.

NOTE: If anyone has a better solution to this issue, please share.

Finally create a visualization of the result to make sure it looks right:

```
python evaluate.py --image_path data/test/test_img.jpg \
                   --heatmap_path data/test/test_heatmap.jpg \
                   --output_name url \
                   --text_preds_path preds/test_json_url.txt
```

Compare this to the local predictions figure to be sure it looks right.


## Batch Prediction

We have not tested this, but if you want to do batch prediction, you should be able to follow the instructions [here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#submit_a_batch_prediction_job).


