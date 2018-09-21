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

In this codebase we are using TensorFlow version 1.10 without a GPU and so we will install it directly using the following command:

pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl

If not on Mac, follow the TensorFlow install instructions [here](https://www.tensorflow.org/install/pip).

## Model Training and Preparation




## General GCP Configuration



## GCP Model Configuration



## Running Prediction









