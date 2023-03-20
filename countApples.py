import json
import os
from PIL import Image
import cv2, numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from colorthief import ColorThief
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot


directory = "videos"


shapesName = "shapes"

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    print("Get the number of different clusters, create histogram, and normalize")
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    print("Create frequency rect and iterate through each cluster's color and percentage")
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

def findFiles(directory):
    currentImage = ""
    for vid in os.listdir(directory):
        f = os.path.join(directory, vid)
        if os.path.isdir(f):
            for file in os.listdir(f):
                g = os.path.join(f,file)
                if not g.endswith(".json"):
                    currentImage = g
                else: 
                    showArea(currentImage, g)
            print("Amount of Apples for Video {0} is {1}".format(file, appleCounter))
            appleCounter = []



def LoadImage(image):
    # Load image and convert to a list of pixels
    imageOld = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
    cv2.imshow('visualize', visualize)
    cv2.imshow("file", imageOld)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showArea(image, label):
    aCounter = 0
    combinedImage = []
    im = Image.open(image)
    f = open(label)
    l = json.load(f)
    for shape in l[shapesName]:
        aCounter = aCounter + 1
        points = shape["points"]
        cropArea = (points[0][0],points[0][1],points[1][0],points[1][1])
        cropedImage = im.crop(cropArea)
        open_cv_image = np.array(cropedImage) 
        # Convert RGB to BGR 
        if open_cv_image.size > 10:
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
        
            #cv2.imshow("",open_cv_image)
            #cv2.waitKey()
            combinedImage.append(open_cv_image)
    appleCounter.append(aCounter)
    print("Amount of apples for image {0} : {1}".format(image,aCounter))
    #LoadImage(open_cv_image)

def calculateRGB(image, label, array, allArray):
    im = Image.open(image)
    f = open(label)
    l = json.load(f)
    for shape in l[shapesName]:
        mean = np.array([0.,0.,0.])
        points = shape["points"]
        cropArea = (points[0][0],points[0][1],points[1][0],points[1][1])
        cropedImage = im.crop(cropArea)
        open_cv_image = np.array(cropedImage) 
        # Convert RGB to BGR 
        if open_cv_image.size > 10:
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            cl = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
            cl = cl.astype(float) / 255
            #fig, axs = plt.subplots(3)
            #axs[0].imshow(cropedImage)
            for j in range(3):
                mean[j] += np.mean(cl[:,:,j])
            allArray.append(mean)
            #axs[1].imshow([[(np.mean(cl[:,:,0]),np.mean(cl[:,:,1]),np.mean(cl[:,:,2]))]])
            #plt.show()
            #cv2.imshow("",open_cv_image)
            #cv2.waitKey()

    array.append(mean/len(l[shapesName]))
    #LoadImage(open_cv_image)

def calculateSTD(image, label, mean, stdTemp):
    im = Image.open(image)
    f = open(label)
    l = json.load(f)
    for shape in l[shapesName]:
        points = shape["points"]
        cropArea = (points[0][0],points[0][1],points[1][0],points[1][1])
        cropedImage = im.crop(cropArea)
        open_cv_image = np.array(cropedImage) 
        # Convert RGB to BGR 
        if open_cv_image.size > 10:
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            cl = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
            cl = cl.astype(float) / 255
            for j in range(3):
                stdTemp[j] += ((cl[:,:,j] - mean[j])**2).sum()/(cl.shape[0]*cl.shape[1])
            #cv2.imshow("",open_cv_image)
            #cv2.waitKey()


def countShapes(image, label, counter):
    aCounter = 0
    im = Image.open(image)
    f = open(label)
    l = json.load(f)
    for shape in l[shapesName]:
        aCounter = aCounter + 1
    counter.append(aCounter)
    #LoadImage(open_cv_image)


def countShapeSize(image, label, array):
    im = Image.open(image)
    f = open(label)
    l = json.load(f)
    for shape in l[shapesName]:
        points = shape["points"]
        width, height = im.size
        pixels = width*height
 
        left = points[0][0]
        upper = points[0][1]
        right = points[1][0]
        lower = points[1][1]

        array.append(abs(right-left)*abs(lower-upper)/pixels)

    #LoadImage(open_cv_image)


def createGraph(image, array, rgbMean, title, allMeans, std):
    roundedArray = []
    countArray = []
    for e in array:
        t = round(e,50)
        if t in roundedArray:
            countArray[roundedArray.index(t)] += 1
        else:
            roundedArray.append(t)
            countArray.append(1)
    #print(roundedArray)
    #print(countArray)
    
    fig, axs = plt.subplots(2,2, figsize=(15,8))
    fig.canvas.manager.set_window_title(title)
    axs[0,0].hist(array, bins=100, range=[np.min(array), np.max(array)], histtype='step',edgecolor='r',linewidth=3)
    print((rgbMean[0],rgbMean[1],rgbMean[2]))
    axs[0,0].gca().yaxis.set_major_formatter(PercentFormatter(1))
    axs[0,1].imshow([[(rgbMean[0],rgbMean[1],rgbMean[2])]])
    axs[1,0].imshow(mpimg.imread(image))

     # Extract the first and second values from each array
    first_vals = np.array([array[0] for array in allMeans])
    second_vals = np.array([array[1] for array in allMeans])
    third_vals = np.array([array[2] for array in allMeans])

    # Compute the frequency distributions for the first and second values
    first_hist = np.histogram(first_vals, bins=60)
    second_hist = np.histogram(second_vals, bins=60)
    third_hist = np.histogram(third_vals, bins=60)

    # Plot the frequency distributions for the first value
    axs[1,1].plot(first_hist[1][:-1], first_hist[0], linewidth=2, label='Red std:{0}'.format(round(std[0],5)), color='r')
    axs[1,1].plot(second_hist[1][:-1], second_hist[0], linewidth=2, label='Green std:{0}'.format(round(std[1],5)), color='g')
    axs[1,1].plot(third_hist[1][:-1], third_hist[0], linewidth=2, label='Blue std:{0}'.format(round(std[2],5)), color='b')
    axs[1,1].set_xlabel('Value')
    axs[1,1].set_ylabel('Frequency')
    axs[1,1].legend()
    plt.show()
    

def countApples(directory):
    currentImage = ""
    for vid in os.listdir(directory):
        appleCounter = []
        sizeCounter = []
        meanCounter = []
        allMeans = []

        f = os.path.join(directory, vid)
        if os.path.isdir(f):
            for file in os.listdir(f):
                g = os.path.join(f,file)
                if not g.endswith(".json"):
                    currentImage = g
                else: 
                    countShapes(currentImage, g, appleCounter)
                    countShapeSize(currentImage, g, sizeCounter)
                    calculateRGB(currentImage, g, meanCounter, allMeans)
            print("Amount of Apples for Video {0} is {1}".format(vid, sum(appleCounter)))
            print("Average Dimension for Video {0} is {1}".format(vid, sum(sizeCounter)/len(sizeCounter)))

            print("Average RGB for Video {0} is {1}".format(vid, sum(allMeans)/len(allMeans)))
            #Calcualte STD
            stdTemp = np.array([0.,0.,0.])
            std = np.array([0.,0.,0.])
            for file in os.listdir(f):
                g = os.path.join(f,file)
                if not g.endswith(".json"):
                    currentImage = g
                else: 
                    calculateSTD(currentImage, g, sum(meanCounter)/len(meanCounter), stdTemp)

            std = np.array([0.,0.,0.])
            std = np.sqrt(stdTemp/sum(appleCounter))
            print("STD: {0} {1}".format(std,stdTemp))
            createGraph(currentImage,sizeCounter,sum(allMeans)/len(allMeans), vid, allMeans, std)
            

#countApples(directory)

def countShapeSizeNew(label, images):
    colors = []
    sizes = []
    f = open(label)
    l = json.load(f)
    currentIndex = None
    im = None
    for a in l["annotations"]:
        mean = np.array([0.,0.,0.])
        if a['image_id'] != currentIndex:
            #Do Some Things
            currentIndex = a['image_id']
            im = Image.open(os.path.join(images,os.listdir(images)[currentIndex]))
        points = a["bbox"]
        cropArea = (points[0],points[1],points[0]+points[2],points[1]+points[3])
        cropedImage = im.crop(cropArea)
        open_cv_image = np.array(cropedImage) 
        # Convert RGB to BGR 
        if open_cv_image.size > 10:
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            
            # Save pixel size in relation to image
            width, height = im.size
            pixels = width*height
            sizes.append(a['area']/pixels)

            cl = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
            cl = cl.astype(float) / 255
            #fig, axs = plt.subplots(3)
            #axs[0].imshow(cropedImage)
            for j in range(3):
                mean[j] += np.mean(cl[:,:,j])
            colors.append(mean)
            #axs[1].imshow([[(mean[0],mean[1],mean[2])]])
            #plt.show()
            
    

    # Plot sizes
    counts, bins, _ = plt.hist(sizes,bins=50)
    percentages = counts / len(sizes) * 100
    plt.clf()  # Clear the previous plot
    fig = plt.figure(figsize=(12, 4))
    axs0 = fig.add_subplot(221)
    axs0.bar(bins[:-1], percentages, width=bins[1]-bins[0])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))  # Format y-axis ticks as percentage
    plt.title(images)

    # Plot Colors

    # Extract the first and second values from each array
    first_vals = np.array([array[0] for array in colors])
    second_vals = np.array([array[1] for array in colors])
    third_vals = np.array([array[2] for array in colors])
    # Compute the frequency distributions for the first and second values
    first_hist = np.histogram(first_vals, bins=60)
    second_hist = np.histogram(second_vals, bins=60)
    third_hist = np.histogram(third_vals, bins=60)

    # Calculate Average 
    r = sum(first_vals)/len(first_vals)
    g = sum(second_vals)/len(second_vals)
    b = sum(third_vals)/len(third_vals)

    axs1 = fig.add_subplot(222)
    axs1.imshow([[(r,g,b)]])

    axs2 = fig.add_subplot(223)
    axs2.plot(first_hist[1][:-1], first_hist[0], linewidth=2, label='Red', color='r')
    axs2.plot(second_hist[1][:-1], second_hist[0], linewidth=2, label='Green', color='g')
    axs2.plot(third_hist[1][:-1], third_hist[0], linewidth=2, label='Blue', color='b')


    axs3 = fig.add_subplot(224, projection="3d") # 3D plot with scalar values in each axis

    axs3.scatter(first_vals, second_vals, third_vals, c=np.array(colors), marker="o")
    axs3.set_xlabel("Red")
    axs3.set_ylabel("Green")
    axs3.set_zlabel("Blue")
    plt.show()

directory = "newVideos"
def count():
    for file in os.listdir(directory):
        f = os.path.join(directory, file)
        if os.path.isdir(f):
            instances = os.path.join(directory, "instances_{0}.json".format(f.split("\\")[1]))
            countShapeSizeNew(instances, f)
        
count()