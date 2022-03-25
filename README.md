# Attendance-System-using-Facial-Recognition

This project was implemented using OpenCV python. Main goal of this project was to develop the attendance system using the face recognition technique and enhance its features to make it efficient for daily usage.

Face detection is a computer vision technology that helps to locate human faces in digital images. This technique is a specific use case of object detection technology that deals with detecting instances of semantic objects of a certain class (such as humans, buildings or cars) in digital images and videos.I have used haarcascade frontal face default classifier to perform the job! . It captures the images in grayscale mode and collects 50 samples like this: 

![image](https://user-images.githubusercontent.com/88980986/160107250-3fe83927-b723-4ba8-94cc-5d8ead888f0a.png)

I have used Histogram Equalization technique to automatically adjust the brightness , contrast and highlights thereby enhancing the details of the image.

![image](https://user-images.githubusercontent.com/88980986/160107438-877b5e1f-d01e-4b6c-bf31-ace6c619acad.png)
![image](https://user-images.githubusercontent.com/88980986/160107406-8e89a8d5-03c3-4918-800e-4bdca9eaa884.png)

A label is assigned to every person and all the pictures of that person are given the same label. Then all the images are converted to NumPy arrays for training. 
After faces are detected, and converted to grayscale images, faces are recognized using the training data. The labels assigned to pictures while training the data are returned and those labels are the indexes of the CMS IDs and names of people recognized! . After that the attendance of students is maintained using the .csv file.



