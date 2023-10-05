import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import csv
from math import sqrt
from math import pow
import time

start_load_model = time.time()

model = keras.models.load_model('E:/ImageSave/new_model/new_model')

end_load_model = time.time()

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def ConvertRange(old_value,old_min,old_max,new_min,new_max):
    return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

def ConvertList(way):
    l = []
    for i in range(len(way)-1, -1,-1):
        l.append(way[i])
    return l

def pointValue(x,y,power,smoothing,xv,yv,values):
    nominator=0
    denominator=0
    for i in range(0,len(values)):
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+smoothing*smoothing)
        #If the point is really close to one of the data points, return the data point value to avoid singularities
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    #print(denominator)
    #Return NODATA if the denominator is zero
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def invDist(xv,yv,values,xsize=100,ysize=100,power=2,smoothing=0):
    valuesGrid = np.zeros((ysize,xsize))
    for x in range(0,xsize):
        for y in range(0,ysize):
            valuesGrid[y][x] = pointValue(x,y,power,smoothing,xv,yv,values)
    #print(valuesGrid)
    return valuesGrid
    

if __name__ == "__main__":
    power=1
    smoothing=20

    xv = [10,60,40,70,10,50,20,70,30,60,10.1,30.2,50.3,80.4,90.5,32.1,42.2,12.3,62.4,1]
    yv = [10,20,30,30,40,50,60,70,80,90,23.1,24.3,25.2,23.2,35.2,68.1,70.1,34.1,68.2,90]
    values = [4,2,2,3,4,3,1,2,1,1,5,3,2,3,2,3,3,1,3,4]

    ZI = invDist(xv,yv,values,64,64,power,smoothing)
    max = ZI[0][0]
    min = ZI[0][0]
    for i in range(64):
        for j in range(64):
            if(ZI[j][i] > max):
                max = ZI[i][j]
            if(ZI[j][i] < min):
                min = ZI[i][j]
    #ZI = ConvertRange(ZI, min, max, 0, 1)

# x = 4*[0]

# Round function

def roundTo(a, maxsize):
    if maxsize == 8:
        if a == 0 or a == 1:
            a = 0
        if a == 2 or a == 3:
            a = maxsize/3
        if a == 4 or a == 5:
            a = 2 * maxsize/3
        if a == 6 or a == 7:
            a = maxsize - 1
        return a

    else:
        if a < maxsize/6:
            a = 0
        elif a >= maxsize/6 and a < maxsize / 2:
            a = maxsize/3
        elif a >= maxsize/2 and a < 5*maxsize / 6:
            a = 2 * maxsize/3
        else:
            a = maxsize - 1
        return a

# Predict and draw the basic line
def predictAndDrawBasicLine(imgInput, tuplePoints):
    (beginX, beginY, endX, endY) = tuplePoints
    img = imgInput.copy()
    # Determine begin and end point
    beginPoint = [beginX, beginY]
    endPoint = [endX, endY]
    (h, d, w) = img.shape
    if h > 4:
        for i in range(0, 2):
            beginPoint[i] = roundTo(beginPoint[i], h)
            endPoint[i] = roundTo(endPoint[i], h)
        coordinate = {"00": [0, 0],
                    "01": [0, d/3],
                    "02": [0, 2*d/3],
                    "03": [0, d-1],
                    "10": [h/3, 0],
                    "11": [h/3, d/3],
                    "12": [h/3, 2*d/3],
                    "13": [h/3, d-1],
                    "20": [2*h/3, 0],
                    "21": [2*h/3, d/3],
                    "22": [2*h/3, 2*d/3],
                    "23": [2*h/3, d-1],
                    "30": [h-1, 0],
                    "31": [h-1, d/3],
                    "32": [h-1, 2*d/3],
                    "33": [h-1, d-1]}
    if h == 4:
        coordinate = {"00": [0, 0],
                    "01": [0, 1],
                    "02": [0, 2],
                    "03": [0, 3],
                    "10": [1, 0],
                    "11": [1, 1],
                    "12": [1, 2],
                    "13": [1, 3],
                    "20": [2, 0],
                    "21": [2, 1],
                    "22": [2, 2],
                    "23": [2, 3],
                    "30": [3, 0],
                    "31": [3, 1],
                    "32": [3, 2],
                    "33": [3, 3]}
    begin = "00"
    end = "00"

    for point in coordinate:
        if coordinate[point] == beginPoint:
            begin = point
        if coordinate[point] == endPoint:
            end = point

    if begin != end:
        imgResize = cv2.resize(img, (4, 4))
        imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)

        # Tạo mảng lower và upper nhằm mục đích sử dụng cho hàm inRange
        lower = np.array([31, 0, 0])
        upper = np.array([255, 255, 255])

        mark = cv2.inRange(imgHSV, lower, upper)

        # Chuyển về màu ban đầu
        imgResult = cv2.bitwise_and(imgResize, imgResize, mask=mark)
        imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)

        imgResize = image.img_to_array(imgResult)
        imgResize = imgResize/255.
        imgResize = np.array([imgResize])

        predict_result = model.predict(imgResize)
        # opening the csv file
        with open('E:\ImageSave\data0.csv') as csv_file:
            # reading the csv file using DictReader
            csv_reader = csv.DictReader(csv_file)
            # converting the file to dictionary
            # by first converting to list
            # and then converting the list to dict
            dict_from_csv = dict(list(csv_reader)[0])
            # making a list from the keys of the dict
            list_of_column_names = list(dict_from_csv.keys())
            # displaying the list of column names
            # print("List of column names : ",list_of_column_names)

        predictList = []

        predict_result = model.predict(imgResize)
        for i in range(112):
            if predict_result[0][i] > 0.8:
                # print(predict_result[0][i])
                predictList.append(list_of_column_names[i].split('-'))
        print("List : ",predictList)
        # Draw a line in img, prioritize path have 3 point
        haveLine = False
        for path in predictList:
            if len(path) == 3 and begin in path and end in path:
                cv2.line(img, (int(coordinate[path[0]][0]), int(coordinate[path[0]][1])), (int(coordinate[path[1]][0]), int(coordinate[path[1]][1])), (255, 0, 0), 1)
                cv2.line(img, (int(coordinate[path[1]][0]), int(coordinate[path[1]][1])), (int(coordinate[path[2]][0]), int(coordinate[path[2]][1])), (255, 0, 0), 1)
                haveLine = True
                print("Point: ", coordinate[path[0]][0])
                break
        if haveLine == False:
            cv2.line(img, (int(coordinate[begin][0]), int(coordinate[begin][1])), (int(coordinate[end][0]), int(coordinate[end][1])), (255, 0, 0), 1)
    return (img, (int(beginPoint[0]), int(beginPoint[1]), int(endPoint[0]), int(endPoint[1])))

# Convert Point
def defineRegion(h, point):
    (x, y) = point
    if (x < h/2):
        xRegion = "Left"
    else:
        xRegion = "Right"
    if (y < h/2):
        yRegion = "Up"
    else:
        yRegion = "Down"
    
    if (xRegion == "Left" and yRegion == "Up"):
        return "img0"
    elif (xRegion == "Right" and yRegion == "Up"):
        return "img2"
    elif (xRegion == "Left" and yRegion == "Down"):
        return "img1"
    else:
        return "img3"

#backTracing @Overwrite
def backTracking(imgInput, tuplePoints):
    (beginX, beginY, endX, endY) = tuplePoints
    imgRaw = imgInput.copy()
    (h, d, w) = imgInput.shape
    if( h == 16 and d == 16):
        return predictAndDrawBasicLine(imgInput, tuplePoints)[0]
    else:
        (imgInput, tuplePoints) = predictAndDrawBasicLine(imgInput, tuplePoints)
        (beginX, beginY, endX, endY) = tuplePoints
        part = {0: [0, int(h/2), 0, int(d/2)],
                1: [int(h/2), int(h), 0, int(d/2)],
                2: [0, int(h/2), int(d/2), int(d)],
                3: [int(h/2), int(h), int(d/2), int(d)]}
        img0 = imgRaw[part[0][0]:part[0][1], part[0][2]: part[0][3]]
        img1 = imgRaw[part[1][0]:part[1][1], part[1][2]: part[1][3]]
        img2 = imgRaw[part[2][0]:part[2][1], part[2][2]: part[2][3]]
        img3 = imgRaw[part[3][0]:part[3][1], part[3][2]: part[3][3]]
        
        listPoints= {"img0" :[], "img1" :[], "img2" :[], "img3" :[]}
        listPoints[defineRegion(h, (beginX, beginY))].append((int(beginX%(d/2)), int(beginY%(h/2))))
        listPoints[defineRegion(h, (endX, endY))].append((int(endX%(d/2)), int(endY%(h/2))))

        # find by x-axis
        for i in range(0, d, 1):
            if all(imgInput[int(h/2)][i] == (255, 0, 0)):
                regionA = defineRegion(h , (i,int(h/2)))
                regionB = defineRegion(h , (i,int(h/2-1)))
                if i == d/2:
                    xA = int((d/2-1)%(d/2))
                    yA = int((h/2-1)%(h/2))
                    xB = int(i%(d/2))
                    yB = int((h/2)%(h/2))
                    listPoints["img0"].append((xA, yA))
                    listPoints[regionA].append((xB, yB))
                    break
                else :
                    xA = int(i%(d/2))
                    yA = int((h/2)%(h/2))
                    xB = int(i%(d/2))
                    yB = int((h/2-1)%(h/2))
                    listPoints[regionA].append((xA, yA))
                    listPoints[regionB].append((xB, yB))
                    break
        # find by y-axis
        for i in range(0, h, 1):
            if all(imgInput[i][int(d/2)] == (255, 0, 0)):
                regionA = defineRegion(h , (int(d/2),i))
                regionB = defineRegion(h , (int(d/2-1),i))
                if i == h/2:
                    break
                else :
                    xA  = int((d/2)%(d/2))
                    yA = int(i%(h/2))
                    xB  = int((d/2-1)%(d/2))
                    yB = int(i%(h/2))
                    listPoints[regionA].append((xA, yA))
                    listPoints[regionB].append((xB, yB))
                    break

        # listPoints = sorted(listPoints,key=lambda d: d['y'])
        listPoints["img0"].sort(key=lambda tup: tup[1])
        listPoints["img1"].sort(key=lambda tup: tup[1])
        listPoints["img2"].sort(key=lambda tup: tup[1])
        listPoints["img3"].sort(key=lambda tup: tup[1])
        if (len(listPoints["img0"]) == 2):
            (xA , yA) = listPoints["img0"][0]
            (xB , yB) = listPoints["img0"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[0][0]:part[0][1], part[0][2]: part[0][3]] = backTracking(img0, pointInput)
        if (len(listPoints["img1"]) == 2):
            (xA , yA) = listPoints["img1"][0]
            (xB , yB) = listPoints["img1"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[1][0]:part[1][1], part[1][2]: part[1][3]] = backTracking(img1, pointInput)
        if (len(listPoints["img2"]) == 2):
            (xA , yA) = listPoints["img2"][0]
            (xB , yB) = listPoints["img2"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[2][0]:part[2][1], part[2][2]: part[2][3]] = backTracking(img2, pointInput)
        if (len(listPoints["img3"]) == 2):
            (xA , yA) = listPoints["img3"][0]
            (xB , yB) = listPoints["img3"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[3][0]:part[3][1], part[3][2]: part[3][3]] = backTracking(img3, pointInput)
    return imgInput

start_time = time.time()
img0 = cv2.imread("E:\ImageSave\Layer2.png")
img = backTracking(img0, (0, 20, 64, 32))
end_time = time.time()
cv2.imwrite("E:\ImageSave\Layer2_path.png", img)


# the draw line is blue
blue = [[255, 0 , 0]]
# white = [[0, 0, 0]]
#browse the array matrix
sum = 0
#matrix 10x10 -> range from 0 to 9
for x in range(64):
    for y in range(64):
        if not((img[x][y]- blue).any()):
            sum += ZI[x][y]

print("Risk path and run time when using model: ")
print("Risk path = ",sum)
print("Time: ", format(end_time - start_time))