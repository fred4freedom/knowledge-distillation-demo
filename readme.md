# Introduction

This is a simple demo of knowledge distillation using the CIFAR-10 dataset. 

# Instructions

## Setup

Poetry must be installed in the python environment:

`pip install poetry`

After which, you can use poetry to set up the project:

`poetry install`

within the project folder

## Train a model directly

Run `python train.py student`

## Train a teacher model first

Run `python train.py teacher`

## Distill the knowledge from a trained teacher into a student model 

Run `python train.py distill`
