# tf-keras-gcloud-deployment
Training and deploying a tf.keras model using Google Cloud ML Engine

## Introduction

While trying to deploy a tf.keras model using Google Cloud Platform (GCP) ML Engine, I ran into many challenges and found the documentation lacking in many ways. Once I finally got things working, I decided to put it all together in this repo to potentially help others who are doing the same thing. This codebase focuses on the cloud deployment side of things. All of the model training is done locally, but there are many tutorials for how to train on the cloud as well. The deployment scenario also mainly deals with image inputs, though handles them in different ways (including a simple list string, a base64 encoding, and GCP Storage link).

What is shown here is just one way to get this whole process working. The specific configurations may not be optimal for everyone's use case. Additionally, this field moves very quickly and the APIs are often changing so what works now may not work in 6 months.

## Setup

This code is tested in python 3.6 on Mac OSX 10.13 with an Anaconda distribution (though all dependencies are reinstalled). The python libraries required are all installed within a virtual environment. If you don't have virtualenv, install it before proceeding. In the project root directory, run the following:

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Setting up TensorFlow

In this codebase we are using TensorFlow version 1.10 without a GPU. We will install it directly using the following command:

```bash
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl
```

If not on Mac, follow the TensorFlow installation instructions [here](https://www.tensorflow.org/install/pip).

## Model Training and Preparation

In this example, we are training a very simple fully convolutional model on a toy dataset. The output heatmaps capture the brightest parts of the image. This problem is not very interesting, but it will allow us to train a model quickly and use it for deployment. The multidimensional nature of the output makes the deployment a bit more interesting, but all the steps below should be the same for a flat output such as what you'd have with a simple classification model.

In the root directory of the repo, run:

```bash
python train.py
```

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

## GCP Model Configuration



## Running Prediction









