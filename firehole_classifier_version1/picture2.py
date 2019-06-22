import cv2
import os
import numpy as np
def mov(file):
	mov=[]
	for i in file:
		j=i.split('.')
		if len(j)>1:
			if j[1]=="avi":
				mov.append(i)
	return mov
def get_huoyan(frame):
	r_img = frame[:,:,2].copy()
	r_img[r_img<230]=0
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	for i in range(3):
		r_img = cv2.erode(r_img,kernel)
	for i in range(3):
		r_img = cv2.dilate(r_img,kernel)
	img_0,contours,hierarchy = cv2.findContours(r_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	huoyan = [x.shape[0] for x in contours]
	huoyan_index =np.argmax(huoyan)
	huoyan = contours[huoyan_index]
	huoyan = huoyan.reshape((-1,2))
	x_min,y_min = huoyan.min(axis =0)
	x_max,y_max = huoyan.max(axis =0)
	# img = frame[y_min:y_max,x_min:x_max,:]
	# return img

	center_x,center_y = int((x_max+x_min)/2),int((y_max+y_min)/2)

	if (center_y-100<0)|(center_y+100>frame.shape[0])|(center_x-100<0)|(center_x+100>frame.shape[1]):
		return []
	else:
		return frame[(center_y-100):(center_y+100),(center_x-100):(center_x+100)]

def write_txt(path,img):
	with open(path,'w') as lines:
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if j!=img.shape[1]-1:
					lines.write(str(img[i,j])+" ")
				else:
					lines.write(str(img[i,j]))
			lines.write("\n")

def picture(file):
	for video in file:
		frame_idx=0
		cap=cv2.VideoCapture(video)
		#num = 0
		l = int(cap.get(7)/(25*2))
		frame_idx=0
		print(video,l)
		for k in range(l):
			num=k*25*2
			#num = 5*25
			data =np.array([])
			data = data.reshape((-1,200))
			data1 =data.copy()
			
			while 1:
				ret,frame=cap.read()
				if frame_idx>=num:
					########高斯模糊########
					frame = cv2.GaussianBlur(frame,(3,3),1.5)
					frame = get_huoyan(frame)
					if len(frame)!=0:

						name=video.split(".")[0]
						cv2.imwrite("./pictures1/"+str(k)+"_"+name+"_"+str(frame_idx-num)+".jpg",frame)
						#cv2.imwrite("./pictures1/"+"1_"+name+"_"+str(frame_idx-num)+".jpg",frame)
						#cv2.imwrite("./pictures1/"+"2_"+name+"_"+str(frame_idx-num)+".jpg",frame)
						#######灰度变换##########
						frame = frame.astype('float')
						frame = cv2.resize(frame,(200,200))
						frame_gray = frame[:,:,0]*0.687+frame[:,:,1]*0.199+frame[:,:,2]*0.114
						data =np.concatenate((data,frame_gray))
						# if frame_idx == num:
						# 	pre = frame_gray.copy()
						# else:
						# 	data1 = np.concatenate((data1,(frame_gray-pre)))
						# 	pre =frame_gray.copy()
					else:
						print(video)
						break
			


				frame_idx+=1
				if frame_idx==num+5:
					#data =np.concatenate((data,data1))
					#data = abs(data)
					#data = data/data.max()
					txt_name=str(k)+"_"+video.split('.')[0]+".txt"
					#txt_name="1_"+video.split('.')[0]+".txt"
					#txt_name="2_"+video.split('.')[0]+".txt"
					write_txt("./txt1/"+txt_name,data)
					break


if __name__=="__main__":
	file=os.listdir()
	print("1")
	file=mov(file)
	print("2")
	picture(file)
