#VFaceAnalyzer

## What is it? 
This project is created as a final project for Aybars Uğur's 2020 Spring Artificial Intelligence class at Ege University Computer Engineering Department.

## What is does? 
Basically, it takes a "The Office" clip from Youtube and creates a detailed report about 
* which characters appeared with what percentage,
* their predominant emotion prediction, 
* average of each characters' age prediction,
* predominant gender prediction of each character etc.

## Where it can be usefeul?
Well, this is a silly application for no particular real life usage. Sole purpose of this application was to show off our DL skills to get a high grade. Which we did. 

## How it works? 
In exact order:
* Takes a frame from the clip and detects and extracts faces with OpenCV HaarCascades(not the most accurate but very fast one).
* Predicts age, emotion and gender of each extracted face. 
* Predicts which The Office character the face belongs.
* Saves the predicted age, gender and emotion for predicted character for report creation.

For age, gender and emotion prediction; we used [@serengil](https://github.com/serengil)'s pre-trained, modified VGG models. Which (kudos to him) work very-very well.

For character classification though, we had to train our own VGG-16 model. We created a The Office cast dataset *by hand*.    


links to required files:
gender model : https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
age model: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
