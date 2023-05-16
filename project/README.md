# Mean shift clustering for image segmentation
## Introduction
This repo is the final project for the course: Data Mining and Machine Learning. Our goal is to write a mean shift clustering algorithm to segment images from scratch without the built-in library and try to optimize it.
## Environment
This project was run on a machine that has:
- numpy=1.23.5
- matplotlib=3.6.3
- opencv=4.5.4
- scipy=1.7.3
- python=3.10

## Project structure

The project is generally structured as follows:
```
root/
├─ project.py
├─ result.png
├─ cow.jpg
```
Here is how to run the script:

	python run project.py

## Result
| Model   | Number of clusters | Time running | Image shape
|----------|------------------------|--------------------|
| Mean shift vanila algorithm | 5178 | 443s | 196 x 300 x 3
| Mean shift vanila algorithm | 2046 | 38s | 98 x 150 x 3
| Mean shift optimization | 5178 | 76s | 196 x 300 x 3
| Mean shift optimization | 2046 | 11s | 98 x 150 x 3

The larger the image, the larger the difference between the running time of the two algorithms.
![result](https://github.com/ssjinkaido/Final-NLP-Project/blob/master/project/result.png)

## Conclusion
The optimized mean shift runs much faster than the vanilla mean shift. Converting the image from RGB to Lab color space is the utmost requirement to speed up the algorithm (multiple times faster, at least 3 times, the speedup ratio will increase when the image size increases), and the segmented image will be much more beautiful.





