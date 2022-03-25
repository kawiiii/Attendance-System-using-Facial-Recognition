import cv2
import os
import sys

name_ = sys.argv[1]
cms_id = sys.argv[2]

# check if folder already exists, if it does, return true else create the folder and return false
def valid_path_check(cms_path, image_directory):
    if not os.path.exists(image_directory):
        os.mkdir(cms_path)
        os.mkdir(image_directory)
        return False
    return True


# Automatic brightness and contrast optimization for enhancing image quality
def image_upscaling(image, clip_hist_percent=1):
    '''
    # Calculate new histogram with desired range and show histogram
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    # gray = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculates histogram in graysclae mode
    cal_histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
    histogram_size = len(cal_histogram)

    # Cumulative Distribution Calculation from Histogram
    Cumulative_distributor = []
    Cumulative_distributor.append(float(cal_histogram[0]))
    for index in range(1, histogram_size):
        Cumulative_distributor.append(Cumulative_distributor[index -1] + float(cal_histogram[index]))

    # Locating Required Points to clip
    maximum = Cumulative_distributor[-1]
    clip_hist_percent *= (maximum/100.0)

    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while Cumulative_distributor[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = histogram_size -1
    while Cumulative_distributor[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values that controls brightness and contrast
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Final Result
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def add_person():
    global cms_id
    global name_
    # creating folder with images
    base_directory = os.path.dirname(os.path.abspath(__file__)) + r"\images"
    cms_path = os.path.join(base_directory, cms_id)
    image_directory = os.path.join(cms_path, name_)

    return name_, cms_path, image_directory


def store_images(image_path):
    # taking and storing 50 images
    count = 0
    while True:

        # Capture video frame
        _, image_frame = vid_cam.read()

        # Convert frame to grayscale
        gray_scale_img = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray_scale_img, 1.3, 5)

        # Loops for each faces
        for (x, y, w, h) in faces:
            # put a rectangle where the face is
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Displays count
            cv2.putText(image_frame, str(count), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow("Frame", image_frame)

            # Save the captured image into the datasets folder
            dataset_folder = image_path + "/" + str(count) + ".png"

            # ------- ENCHANCING IMAGE QUALITY --------
            cv2.imwrite(dataset_folder, gray_scale_img[y:y + h, x:x + w])
            image = cv2.imread(dataset_folder)
            auto_result, alpha, beta = image_upscaling(image)
            auto_result_grayscaled = cv2.cvtColor(auto_result, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dataset_folder, auto_result_grayscaled)

        # To stop taking video, press 'q' for at least 100ms
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        elif count >= 50:
            print("Successfully Captured and Stored Images ! ")
            break


# update images of an existing cms id
# if the person who is already registered is added, this function will ask if they want to update the photos stored
def update_images():
    if valid_path_check(cms_folder_path, img_dir):
        update_imgs = input("CMS ID already registered\nDo you want to update images?")
        if update_imgs.lower() == "yes":
            # image_folder = os.path.dirname(os.path.abspath(__file__)) + r"\images"
            # deleting old photos in the folder to store new ones
            for root, dirs, files in os.walk(img_dir):
                for pictures in files:
                    os.unlink(os.path.join(root, pictures))
            store_images(img_dir)
    else:
        store_images(img_dir)


# creating the images folder if it doesn't exist
images_directory = os.path.dirname(os.path.abspath(__file__)) + r"\images"
if not os.path.exists(images_directory):
    os.mkdir(os.path.dirname(os.path.abspath(__file__)) + r"\images")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid_cam = cv2.VideoCapture(0)
# calling the add person and the update images functions
name, cms_folder_path, img_dir = add_person()
update_images()
