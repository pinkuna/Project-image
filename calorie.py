import cv2
import numpy as np
from image_segment import *

#density - gram / cm^3
density_dict = { 1:0.96, 2:0.95, 3:0.72, 4:0.39, 5:0.814, 6:0.47, 7:0.47, 8:0.64} 
#kcal
calorie_dict = { 1:52.1, 2:88.7, 3:96, 4:66.9, 5:53.3, 6:39.7, 7:17.7, 8:30.4}
#skin of photo to real multiplier
skin_multiplier = np.pi*1*1

def getCalorie(label, volume): #volume in cm^3
	calorie = calorie_dict[int(label)]
	density = density_dict[int(label)]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
	if  label == 1 or label == 4 or label == 5 or label == 6 or label == 7 or label == 8: #sphere-apple,tomato,orange,kiwi,onion
		radius = 3.75#np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		print (area_fruit, radius, volume, skin_area)
	
	if label == 2 or (label == 3 and area_fruit > 30): #cylinder like banana, cucumber, carrot
            fruit_rect = cv2.minAreaRect(fruit_contour)
            height = max(fruit_rect[1])*pix_to_cm_multiplier
            radius = area_fruit/(2.0*height)
            volume = np.pi*radius*radius*height
            print(height, radius)
	
	return volume

def detect_coin(image):
    output = image.copy()

    data=os.path.join(os.getcwd(),"images")
    if os.path.exists(data):
        print('folder exist for images at ',data)
    else:
        os.mkdir(data)
        print('folder created for images at ',data)

    cv2.imwrite('{}\\17 original image.jpg'.format(data),output)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}\\18 gray.jpg'.format(data),gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite('{}\\19 blurred.jpg'.format(data),blurred)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                            param1=200, param2=100, minRadius=5, maxRadius=70)

    count = 0
    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        x, y, d = (circles[0])
        print(d)
        roi = image[y - d:y + d, x - d:x + d]
        if False:
            m = np.zeros(roi.shape[:2], dtype="uint8")
            w = int(roi.shape[1] / 2)
            h = int(roi.shape[0] / 2)
            cv2.circle(m, (w, h), d, (255), -1)
            maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
            cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)

        cv2.circle(output, (x, y), d, (255, 0, 0), 3)
        cv2.imwrite('{}\\20 detect_coin.jpg'.format(data),output)
        cv2.putText(output, '10 bath',
                    (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("cir", output)
        cv2.waitKey(0)
        return d
    else:
        return 1

def calories(result,img):
    img_path =img 

    fruit_areas,final_f,areaod, fruit_contours = getAreaOfFood(img_path)
    radius = detect_coin(img_path)
    skin_areas = np.pi*radius*radius
    pix_cm = (1.3/radius)
    print(radius, pix_cm, skin_areas)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    fruit_volumes=volume
    fruit_calories=cal
    fruit_calories_100grams=cal_100
    fruit_mass=mass
    print("\nfruit_volumes",fruit_volumes,"\nfruit_calories",fruit_calories,"\nfruit_calories_100grams",fruit_calories_100grams,"\nfruit_mass",fruit_mass)
    return fruit_calories

if __name__ == '__main__':
    
    a='4.jfif'
    a=cv2.imread(a)
