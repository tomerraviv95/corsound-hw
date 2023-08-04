# Use the official Python 3.7 image as the base image
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /python-code

# Copy your Python code into the container
COPY . /python-code