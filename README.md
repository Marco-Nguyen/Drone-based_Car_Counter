# Drone-based_Car_Counter

**My contact:**
<p>
  <a href="https://www.linkedin.com/in/hoang-nguyen-tai-93557120a/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
  </a>
</p>  

This is my project for  [Drone Surveillance Contest](https://www.computervision.zone/dsc/?contest=contest-condition)  
Here are the steps I used in this project:  
1. Train YOLOv4 for custom dataset. Here I used frames extracted from the video. If you want to access to the dataset, please contact me.  
2. Once you get the weights, infer it to the video (without counting) to test the weights. My weights achieve up to 98% accuracy. The code to infer the weights is Car_prediction_(no_counter).py  

3. For the counter part, I find an useful technique called "Deepsort". I cloned the work of [theAIGuys](https://github.com/theAIGuysCode/yolov4-deepsort). Check out his guide to implement to your project.  
4. As I have my own custom weights, I used this [link](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) to change to my weights.    
5. If you have trouble finding the correct code to use, here what I used.  
```
# For custom weights
python save_model.py --weights ./data/custom.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 
#For writing videos
python object_tracker.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/results.avi
```
6. I also change a few lines of code for a better results. Find out yourself in object_tracker_custom.py  

Thanks for visiting my repos. Have a nice day. 

P/S: I am finding a way to have DeepSort repos in this project. Some files exceed the limit.
