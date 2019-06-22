import pandas as pd
from glob import glob
import numpy as np
import cv2
import imageio
from tqdm import tqdm
def gethy(frame):
	frame1 = frame.copy()
	frame[frame<230] = 0
	frame[frame!=0] = 255
	frame = np.mean(frame,axis = 2)
	frame = frame.astype(np.uint8)
	img_0, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	length = [len(contour) for contour in contours]
	ind = length.index(max(length))
	contour = contours[ind]
	contour = contour.reshape((-1,2))
	minx = min(contour[:,0])
	miny = min(contour[:,1])
	maxx = max(contour[:,0])
	maxy = max(contour[:,1])
	return minx,miny,maxx,maxy
def get_data():
	label = pd.read_csv("label.csv")
	label['target'] = label['label'].map({"cold":0,'normal':1,'hot':2})
	label_dict = dict(zip(label['Num'],label['target']))
	paths = glob("*.avi")
	n = 10
	span = 5*25
	frame_size = 100
	train = np.array([]).reshape((0,n,frame_size,frame_size,3))
	label = []
	for path in tqdm(paths):
		cap = cv2.VideoCapture(path)
		path_save = "data/"+path.split(".")[0]
		ind_save = 0
		flag_start = 0
		ind_start = 0
		ind = 0
		frames = np.array([]).reshape((0,frame_size,frame_size,3))
		while True:
			ret,frame = cap.read()
			if not ret:
				break
			if flag_start ==0:
				minx,miny,maxx,maxy = gethy(frame)
				cen_x,cen_y =(minx+maxx)//2,(miny+maxy)//2
				if cen_x<frame_size//2 or cen_x+frame_size//2>=frame.shape[1] or cen_y<frame_size//2 or cen_y+frame_size//2>=frame.shape[0]:
					continue
				frames = np.concatenate([frames,np.expand_dims(frame[cen_y-frame_size//2:cen_y+frame_size//2,cen_x-frame_size//2:cen_x+frame_size//2],axis =0)],axis = 0)
				out = cv2.VideoWriter(path_save+"_"+str(ind_save)+".avi",cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_size,frame_size))
				flag_start = 1
				ind_start = ind
			else:
				if ind_start+n-1<ind<ind_start+n+span:
					ind+=1
					continue
				elif ind>=ind_start+n+span:
					train = np.concatenate([train,np.expand_dims(frames,axis = 0)])
					frames = np.array([]).reshape((0,frame_size,frame_size,3))
					label.append(label_dict[path[:-3]+"txt"])
					ind_save +=1
					flag_start = 0
				else:
					frames = np.concatenate([frames,np.expand_dims(frame[cen_y-frame_size//2:cen_y+frame_size//2,cen_x-frame_size//2:cen_x+frame_size//2],axis =0)],axis = 0)
					# out.write(frame[cen_y-frame_size//2:cen_y+frame_size//2,cen_x-frame_size//2:cen_x+frame_size//2])
			ind+=1
		out.release()
		cv2.destroyAllWindows()
	return train,label
if __name__ =="__main__":
	train,label = get_data()

