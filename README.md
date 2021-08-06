# Superpoint libtorch c++ 

-----------------------------------------------------------
## Introduction

-----------------------------------------------------------

This repo contains the source code implemented SuperPoint with libtorch c++ (version 1.3.0 or 1.8.1)

If you want use c++ 11 version. you only choose libtorch 1.3.0

## prerequisite

you must have cuda

## How to use

1. git clone
2. ./download_weight_torch.sh
3. mkdir build
4. cd ./build
5. If you want use c++ 11 then  <code>cmake -DCXX_VERSION=11 </code> else <code> -DCXX_VERSION=1x </code>   
6. make
7. cd ..
8. run ./main

## use your images
use sample folder

## reference
- https://github.com/magicleap/SuperPointPretrainedNetwork
- https://github.com/KinglittleQ/SuperPoint_SLAM

## next step
to be ros sample.
