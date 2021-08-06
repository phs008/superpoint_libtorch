# Superpoint libtorch c++ 

-----------------------------------------------------------
## Introduction

-----------------------------------------------------------

This repo contains the source code implemented SuperPoint with libtorch c++ (version 1.3.0 or 1.8.1)

If you want use c++ 11 version. you only choose libtorch 1.3.0

## prerequisite

you must have cuda


## How to use

-- Plz use Clion 

1. git clone
2. download pre_weight from [here](https://soongsilac-my.sharepoint.com/:u:/g/personal/phs008_soongsil_ac_kr/ERE4KfrXD3RBknZ9dFGcOYUBka5t5Kr1wf5SrRRp8mlSXg?e=xMqrE5)
3. mkdir Thirdparty
4. cd Thirdparty
5. download libtorch version which you want [libtorch 1.3.0](https://soongsilac-my.sharepoint.com/:u:/g/personal/phs008_soongsil_ac_kr/EdtuAaBaY6pFsjHjxM27YYIBeIov7aDFRI8Bx-D86lU9Ig?e=5a73oS) , [libtorch 1.8.1](https://soongsilac-my.sharepoint.com/:u:/g/personal/phs008_soongsil_ac_kr/EVWhU-_p2RZHsJKnVVjaWYQBO0Q-P3SydZesHypPyxXD0w?e=pVgnqN)
6. unzip 
7. If you want use c++ 11 then  <code>-DCXX_VERSION=11 </code> else <code> -DCXX_VERSION=1x </code>   


## use your images
use sample folder

## sample
![sample](https://user-images.githubusercontent.com/17635409/128466926-3afb7bfb-265f-4f60-aa36-3daeac1c0434.gif)

Geforce RTX 2070 , 2 image , avg 250 ms~300 ms (torch 1.8.1)

## reference
- https://github.com/magicleap/SuperPointPretrainedNetwork
- https://github.com/KinglittleQ/SuperPoint_SLAM

## next step
to be ros sample.
