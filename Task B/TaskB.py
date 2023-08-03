import cv2
import cv2.aruco as aruco
import numpy as np


def rotate_image(image, angle, scale):
  
   image = cv2.bitwise_not(image)
   image_center = tuple(np.array(image.shape[1::-1]) / 2)
   rot_mat = cv2.getRotationMatrix2D(image_center, -angle, scale)
   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
   result = cv2.bitwise_not(result)

   return result

def mask_aruco(aruco_list):
   mask = []
   for i in aruco_list:
      ar_img = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

      i = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
      ar_img = cv2.Canny(ar_img,20,70)
      contours, _ = cv2.findContours(ar_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cv2.fillPoly(i,[contours[0]],(0,0,0))
      i = cv2.bitwise_not(i)
      x, y, w, h = cv2.boundingRect(contours[0])
      i = i[y:y+h,x:x+w]
      # cv2.imshow("img",i)
      # cv2.waitKey(0)
      mask.append(i)
   return mask

def overlay_aruco(src1,src2,mask):
   
   modified_aruco_mask = []
   c = 0
   for i in src2:
               
      img = i
      src2_Can = cv2.Canny(i,20,70)
      cnt,_ = cv2.findContours(src2_Can,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      x,y,w,h = cv2.boundingRect(cnt[0])
      img = img[y:y+h,x:x+w]

      tmp_mask = np.zeros(img.shape[1::-1], dtype="uint8")
      tmp_mask = cv2.bitwise_and(img,img,mask = mask[c])
      modified_aruco_mask.append(tmp_mask)
      # cv2.imshow('img',tmp_mask)
      # cv2.waitKey(0)
      c +=1
      


   src1_gray = cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY)

   ret,thresh = cv2.threshold(src1_gray,225,255,0) # Doubt about second parameter
   contours_src1, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   c = 0
   for cnt in contours_src1:
      approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
   
      if len(approx) == 4:
         x, y, w, h = cv2.boundingRect(cnt)
         
         ratio = w/h
         if ratio >= 0.9 and ratio <= 1.1:
            img = src1[y:y+h,x:x+w]
            mask1 = cv2.bitwise_not(mask[c])
            modified2_mask = modified_aruco_mask[c]
            if c == 1:
               mask1 = np.pad(mask1,((1,0),(1,0)),mode = 'constant',constant_values=255)
               modified2_mask = np.pad(modified_aruco_mask[c],((1,0),(1,0),(0,0)),mode = 'minimum')
            img = cv2.bitwise_and(img,img,mask = mask1)
            
            print(mask1.shape,modified2_mask.shape,img.shape)
            
            img =  cv2.add(img,modified2_mask)
            src1[y:y+h,x:x+w] =img
            c += 1
   return src1
            
            

taskImg = cv2.imread("CVTask.png",cv2.IMREAD_GRAYSCALE)
# taskImg = cv2.resize(taskImg,(0,0),fx=0.7,fy=0.7)
taskImg_BGR = cv2.imread("CVTask.png")
# taskImg_BGR = cv2.resize(taskImg,(0,0),fx=0.7,fy=0.7)

angle_sqrs = []
angle_aruco = []
area_squres = []
area_aruco = []

# task_mask = np.zeros(taskImg.shape[:2], dtype="uint8")

ret,thresh = cv2.threshold(taskImg,225,255,0) # Doubt about second parameter
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(taskImg_BGR, contours, -1, (255,0,0), 3)
# print(taskImg.shape)

for cnt in contours:

   # The function cv2.approxPolyDP approximates a curve or a polygon with another curve/polygon
   # with less vertices so that the distance between them is less or equal to the specified precision.
   # The function cv2.arcLength computes a curve length or a closed contour perimeter.

   approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
  
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      temp_angle_list = []
      ratio = w/h
      if ratio >= 0.9 and ratio <= 1.1:
         # print(approx[0][0],approx[1][0],approx[2][0],approx[3][0])
         # taskImg_BGR = cv2.drawContours(taskImg_BGR, [cnt], -1, (255,0,0), 3)       
         # cv2.putText(taskImg_BGR, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
         # m = (approx[1][0][1] - approx[0][0][1]) /(approx[1][0][0] - approx[0][0][0]) 
         _,__,angle_with_x = cv2.minAreaRect(cnt)
         angle_sqrs.append(angle_with_x)
         img = taskImg_BGR[y:y+h,x:x+w]
         tmp_area = cv2.contourArea(cnt) 
         area_squres.append(tmp_area)
         cv2.rectangle(taskImg_BGR, (x, y), (x+w, y+h), (255, 0, 0), 2)
         
         # task_mask = cv2.fillPoly(task_mask,[cnt],(255,255,255)) # Creating mask of the task img
         # cv2.imshow("img",task_mask)
         # cv2.waitKey(0)
# task_mask = cv2.bitwise_not(task_mask)
# cv2.imshow("img",task_mask)
# cv2.waitKey(0)

print(area_squres)
# black = angle_sqrs[0], grey = angle_sqrs[1] , orange = angle_sqrs[2], green = angle_sqrs[3]
      
print(angle_sqrs)
# cv2.imshow('img',taskImg_BGR)
# cv2.waitKey(0)


# Aruco Detector

arucos  = []
arucos_with_ids = []

arucos.append(cv2.imread("Ha.jpg"))
arucos.append(cv2.imread("HaHa.jpg"))
arucos.append(cv2.imread("XD.jpg"))
arucos.append(cv2.imread("LAMO.jpg"))


# A dictionary containing predefined markers dictionaries/sets is created.
# Each dictionary indicates the number of bits and the number of markers contained.

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
params = aruco.DetectorParameters()

arucoDetector = aruco.ArucoDetector(aruco_dict,params)

for img in arucos:
    
    corners, ids, rejectedImagePoints = arucoDetector.detectMarkers(img)
    # print(ids)
    if ids is not None:
            # tmp_img = cv2.GaussianBlur(img,(3,3),0)
            id = ids[0][0]
            tmp_img = cv2.Canny(img,20,70)
            cnt,_ = cv2.findContours(tmp_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img,cnt,-1,(255,0,0),2)
            tmp_area = cv2.contourArea(cnt[0]) 
            # print(tmp_area)
            area_aruco.append(tmp_area)
            _,__,angle_with_x = cv2.minAreaRect(cnt[0])
            angle_aruco.append(angle_with_x)
            # print(ids)
            # cv2.imshow("img",img)
            # cv2.waitKey(0)
            # img = cv2.putText(img, str(id),
                           #  tuple(corners[0][0][0].astype('int')),
                              #   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            arucos_with_ids.append((img,id))
            
scale = np.divide(area_squres,area_aruco) ** 0.5
# print(scale)

# arucos_with_ids = [(img1,id1),(img2,id2)]

final_aruco = []
c = 0

for i in arucos_with_ids:
   rot_angle = 0
   if i[1] == 1:
      rot_angle = angle_sqrs[3] - angle_aruco[3]  # green
   elif i[1] == 2:
      rot_angle = angle_sqrs[2] - angle_aruco[2]   #  orange
   elif i[1] == 3:
      rot_angle = angle_sqrs[0] - angle_aruco[0]  # black
   elif i[1] == 4:
      rot_angle = angle_sqrs[1] - angle_aruco[1]  # grey
   else:
      rot_angle = 0

   rotated_image = rotate_image(i[0],rot_angle,scale[c])
   c += 1
   final_aruco.append(rotated_image)
   # cv2.imshow("imgg",rotated_image)
   # cv2.waitKey(0)




mask_aruco_list = mask_aruco(final_aruco)
# cv2.imshow("imgg",mask_aruco_list[0])
# cv2.waitKey(0)
res_img = cv2.imread("CVTask.png")
result = overlay_aruco(res_img,final_aruco,mask_aruco_list)
cv2.imwrite("Result.jpg",result)

