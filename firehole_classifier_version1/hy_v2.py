from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  
import cv2  
from time import clock  
import csv
import os
import time
  
lk_params = dict( winSize  = (15, 15),   
                  maxLevel = 2,   
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))      
  
feature_params = dict( maxCorners = 500,   
                       qualityLevel = 0.3,  
                       minDistance = 7,  
                       blockSize = 7 )  

def getNeedTrack(img, tracks, starts):
    img_blur = cv2.blur(img, (20,20))
    img_blur[img_blur<=230] = 0
    img_blur[img_blur>230] =255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))  
    img_erode = cv2.erode(img_blur, kernel)
    img_dilate = cv2.dilate(img_erode,kernel) 
    img_dilate = cv2.dilate(img_dilate,kernel)
    img_0, contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #x_arr, y_arr = np.where(img_dilate == 255)
    shape_arr = [i.shape[0] for i in contours]
    shape_max = np.argmax(shape_arr)
    positions_arr = contours[shape_max].reshape(-1, 2)
    #positions = zip(x_arr, y_arr)
    #positions_arr = np.array(positions)
    x_min, y_min = positions_arr.min(axis = 0)
    x_max, y_max = positions_arr.max(axis = 0)
    x_central = np.round((x_min+x_max)/2.0)
    y_central = np.round((y_min+y_max)/2.0)
    height = (y_max - y_min)/2
    width = (x_max - x_min)/2
    diff_arr = [(tr[0][0] - x_central)**2 + (tr[0][1] - y_central)**2 for tr in tracks]
    diff_arr = np.array(diff_arr)
    #needed_tr = tracks[np.argmin(diff_arr)]
    k = np.where(np.array(starts) == min(starts))[0]

    error_ind = np.argmax(diff_arr[k])
    error_tr = tracks[error_ind]
    #error_tr_diff = [[error_tr[i+1][0]-error_tr[i][0], error_tr[i+1][1]-error_tr[i][1]] for i in range(len(error_tr)-1)]
    x_central_final = x_central + error_tr[-1][0] - error_tr[0][0]
    y_central_final = y_central + error_tr[-1][1] - error_tr[0][1]
    diff_arr_final = [(tr[-1][0] - x_central_final)**2 + (tr[-1][1] - y_central_final)**2 for tr in tracks]
    needed_ind = np.argmin(diff_arr_final)
    needed_start = starts[needed_ind]
    needed_tr = tracks[needed_ind]
    #print([x_central, y_central])
    cv2.circle(img, (int(x_central), int(y_central)), 5, 0, 3)
    cv2.circle(img, (int(error_tr[-1][0]), int(error_tr[-1][1])), 5, 0, 3)
    cv2.circle(img, (int(needed_tr[-1][0]), int(needed_tr[-1][1] )), 5, 0, 3)
    return needed_tr, error_tr, [x_central, y_central], [width, height]

class App:  
    def __init__(self, video_src):
        self.track_len = 10  
        self.detect_interval = 5  
        self.tracks = []  
        self.cam = cv2.VideoCapture(video_src)  
        self.frame_idx = 0 
        self.r_img = [] 
        self.starts = []
        self.imgs = []
        self.rimgs = []
        self.color_frame = []
        self.rgb =[]
        self.lost_track = []
  
    def run(self): 
        while True:  
            ret, frame = self.cam.read()
            frame = np.rot90(frame, 3)
            #if self.frame_idx == 0:
             #   vis_col = frame.copy()
              #  b, g, r = cv2.split(vis_col)
               # index_x, index_y = np.where(r >= 240)
                #index = zip(index_x, index_y)
            num =5*29
            if ret == True:  
                if self.frame_idx>=num:

                    b, g, r = cv2.split(frame)
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    vis = frame.copy()
                    vis_gray = frame_gray.copy()  
                    if len(self.tracks) > 0:
                        img0, img1 = self.prev_gray, frame_gray 
                        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)  
                        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                        d = abs(p0-p0r).reshape(-1, 2).max(-1)
                        #print(p0)
                        #print('#################')
                        #print(p0r)
                        #print('########################')
                        #good = d < 1
                        good = d < 1
                        #print(good)
                        new_tracks = []  
                        new_starts = []
                        for tr, (x, y), good_flag, s in zip(self.tracks, p1.reshape([-1, 2]), good, self.starts):
                            if not good_flag:
                                self.lost_track.append(tr)
                                continue  
                            tr.append((x, y))  
                            #if len(tr) > self.track_len:  
                            #    del tr[0]  
                            new_tracks.append(tr)  
                            new_starts.append(s)
                            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)  
                        self.tracks = new_tracks  
                        self.starts = new_starts
                        #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                        #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))  
                    if self.frame_idx % self.detect_interval == 0:
                    #if self.frame_idx == 0:
                        mask = np.zeros_like(frame_gray)
                        mask[:] = 255
                        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                            cv2.circle(mask, (x, y), 5, 0, -1)  
                        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                        #print(p.shape)
                        #print(p.shape)
                        p2 = np.array([tr[-1] for tr in self.tracks]).reshape(-1, 2)
                        #if ( len(p) != 0) & (len(p2) == 0):
                        #    for x, y in np.float32(p).reshape(-1, 2):
                        #        self.tracks.append([(x, y)])
                        #if ( len(p) != 0) & (len(p2) != 0):
                        if (p is not None) & (p2.size == 0):
                            for x, y in np.float32(p).reshape(-1, 2):
                                self.tracks.append([(x, y)])
                                self.starts.append(self.frame_idx)
                            #print(np.asarray(self.tracks).shape)
                        elif (p is not None) & (p2.size != 0):  
                            for x, y in np.float32(p).reshape(-1, 2):  
                                #self.tracks.append([(x, y)])
                                if self.frame_idx == 0:
                                    self.tracks.append([(x, y)])
                                    self.starts.append(self.frame_idx)
                                    print("############")
                                    continue
                                p_diss = abs(p2 - np.array([x, y]))
                                p_distance = [(_p[0]**2+_p[1]**2) for _p in p_diss]
                                if min(p_distance) > 7:
                                    self.tracks.append([(x, y)])
                                    self.starts.append(self.frame_idx)
                        #if (p is not None) & (len(p2) == 0):
                        #    for x, y in np.float32(p).reshape(-1, 2):
                        #        self.tracks.append([(x, y)])
                                #else min(p_distance) < 7:
                                #    index = np.argmin(p_distance)
                                #    self.tracks[int(index)][-1] = (x, y)

                        #print(p)
                        #for track in self.tracks:
                        #    x, y = track[-1]
                        #    point_features = cv2.circle(vis_gray, (x, y), 5, 0, -1)
                        #    point_features = cv2.putText(vis_gray, str((int(x), int(y))), (x,y), 0, 0.5, (0,0,255),2)
                        #cv2.imwrite('hy_1.jpg', point_features)
                        #cv2.imshow('first_frame', point_features)
                        #cv2.waitKey (0)  
                        #cv2.destroyAllWindows()
                    if self.frame_idx % 29 == 0:
                        self.rimgs.append(r)
                        self.imgs.append(frame_gray)
                        self.rgb.append(frame)
                        self.color_frame.append(self.frame_idx-num)
                    self.prev_gray = frame_gray  
                self.frame_idx += 1  
                #cv2.imshow('lk_track', vis) 
                if self.frame_idx == num+1:
                    b, g, r = cv2.split(frame)
                    self.r_img = r.copy()
                    self.img = frame
                    #break 
                if self.frame_idx == 29*10+num:
                    break
        #print(self.tracks[0][-1])
        new_tracks = []
        new_starts = []
        for tr, st in zip(self.tracks, self.starts):
            if len(tr) > 29:
                new_tracks.append(tr)
                new_starts.append(st)
        self.tracks = new_tracks
        self.starts = new_starts

            #ch = 0xFF & cv2.waitKey(1)  
            #if ch == 27:  
                #break  

def get_pictures(imgs, ps, radius, name):
    num = len(imgs)
    for i in range(num):
        img = imgs[i]
        x = int(ps[i][0])
        y = int(ps[i][1])
        img = cv2.rectangle(img, (x-radius, y-radius), (x+radius, y+radius), (255,255,255), 2)
        cv2.imwrite('tupian/'+name.split('.')[0]+'_%s.jpg'%(str(i)), img)

def features(video_path):
    try: video_src = video_path  
    except: print('Please input video name!')
    #print __doc__  
    #App(video_src).run()  
    #cv2.destroyAllWindows() 
    hy = App(video_src)
    hy.run()
    all_tracks = hy.tracks
    #frames = len(all_tracks[0])
    #f = open('232.csv', 'wb')
    #writer = csv.writer(f)
    #writer.writerows(all_tracks)
    #f.close()

    #tr_hy, tr_error = getNeedTrack(hy.r_img, all_tracks)
    tr_hy, tr_error, central, w_h = getNeedTrack(hy.r_img, all_tracks, hy.starts)
    #cv2.circle(hy.img, (int(central[0]), int(central[1])), 5, 0, 3)
    #cv2.imwrite('3.jpg', hy.img)
    optical_flows = flow_features(tr_hy, tr_error)
    if len(optical_flows)==4:
        ps = get_hy_ps(central, tr_error, hy.color_frame)
        get_pictures(hy.imgs, ps, int(w_h[1]), os.path.basename(video_path))
        #color_v = color_features(hy.imgs, ps, int(w_h[1]))
        #color_r = color_features(hy.rimgs, ps, int(w_h[1]))
        color = color_feature_v2(hy.rgb, ps, int(w_h[1]))
    else:
        color=[]

    return optical_flows, color,hy

    #for track in hy.tracks:
    #    x, y = track[-1]
    #    point_features = cv2.circle(hy.prev_gray, (x, y), 5, 0, -1)
    #    point_features = cv2.putText(hy.prev_gray, str((int(x), int(y))), (x,y), 0, 0.5, (0,0,255),2)
    #cv2.imwrite('hy_0.jpg', point_features)
    #cv2.imshow('first_frame', point_features)
    #cv2.waitKey (0)  
    #cv2.destroyAllWindows()
    #index = 0
    #print(len(all_tracks))
    #for track in all_tracks:
    #    if len(track) != 250:
    #        del all_tracks[index]
    #    index += 1
    #print(len(all_tracks))
    #print([len(all_tracks[i]) for i in range(len(all_tracks))])
    #print(all_tracks[10])

def get_hy_ps(central, tr_error, frames):
    ps = [central]
    for i in range(1, len(frames)):
        error_x = tr_error[frames[i]][0]-tr_error[0][0]
        error_y = tr_error[frames[i]][1]-tr_error[0][1]
        central_x = central[0]+error_x
        central_y = central[1]+error_y
        ps.append([central_x, central_y])
    return ps

def color_feature_v2(imgs, ps, radius):
    num = len(imgs)
    cms = []
    css = []
    for i in range(num):
        img = imgs[i]
        x = int(ps[i][0])
        y = int(ps[i][1])
        y_1 = y-radius
        y_2 = y+radius
        x_1 = x-radius
        x_2 = x+radius

        if y_1 <0:
            y_1 = 0
        if y_2 >=img.shape[0]:
            y_2 = img.shape[0]-1
        if x_1<0:
            x_1 = 0
        if x_2>=img.shape[1]:
            x_2 = img.shape[1]-1
        img = img[y_1:y_2, x_1:x_2, :]
        img = np.max(img, 2)
        cms.append(img.mean())
    v_mean = [cms[i+1]-cms[i] for i in range(len(cms)-1)]

    v_mean_mean = np.mean(v_mean)
    print(v_mean_mean)
    return v_mean_mean

def color_features(imgs, ps, radius):
    num = len(imgs)
    cms = []
    css = []
    for i in range(num):
        img = imgs[i]
        x = int(ps[i][0])
        y = int(ps[i][1])
        img = img[(y-radius):(y+radius), (x-radius):(x+radius)]
        img = img.flatten()
        color_mean = np.mean(img)
        color_std = np.std(img)
        cms.append(color_mean)
        css.append(color_std)
    v_mean = [cms[i+1]-cms[i] for i in range(len(cms)-1)]
    v_std = [css[i+1]-css[i] for i in range(len(css)-1)]

    v_mean_mean = np.mean(v_mean)
    v_std_mean = np.mean(v_std)
    return v_mean_mean, v_std_mean

def flow_features(tr_hy, tr_error):
    #frame_color = []
    #print(len(tr_hy))
    #print(len(tr_error))
    frames_hy = len(tr_hy)
    frames_error = len(tr_error)
    assert frames_hy <= frames_error
    frame_diff = frames_error - frames_hy

    diff_arr = []
    #print(track_2[0])
    for i in range(frames_hy - 1):
        x0, y0 = tr_hy[i]
        next_x0, next_y0 = tr_hy[i+1]
        diff_x0 = next_x0 - x0
        diff_y0 = next_y0 - y0

        x2, y2 = tr_error[i+frame_diff]
        next_x2, next_y2 = tr_error[i+1+frame_diff]
        diff_x2 = next_x2 - x2
        diff_y2 = next_y2 - y2

        diff_x = diff_x0 - diff_x2
        diff_y = diff_y0 - diff_y2
        #print((np.floor(diff_x), np.floor(diff_y)))
        diff_arr.append((np.floor(diff_x), np.floor(diff_y)))
    #print(diff_arr)
    num = len(diff_arr)
    index = []
    #print(num)
    j = 0
    for i in range(num):
        if i == 0:
            previous = diff_arr[i][-1]
            continue
        if (previous * diff_arr[i][-1] <0)|((previous!=0)and(diff_arr[i][-1]==0))|((previous==0) and (diff_arr[i][-1]!=0)) :
            index.append(i)
            j += 1

        previous = diff_arr[i][-1]
    #print(index, j)
    #frame_color = [i+frame_diff for i in index]


    dis_arr = []
    frame_arr = []
    for i in range(len(index) - 1):
        real_dis_x = abs((tr_hy[index[i]][0] - tr_hy[index[i+1]][0]) - (tr_error[index[i]+frame_diff][0] - tr_error[index[i+1]+frame_diff][0]))
        real_dis_y = abs((tr_hy[index[i]][-1] - tr_hy[index[i+1]][-1]) - (tr_error[index[i]+frame_diff][-1] - tr_error[index[i+1]+frame_diff][-1]))
        real_dis = np.sqrt(real_dis_x**2 + real_dis_y**2)
        real_frame = index[i+1] - index[i]
        dis_arr.append(real_dis)
        frame_arr.append(real_frame)
    #print(dis_arr, frame_arr)
    max_dis = max(dis_arr)
    #max_index = np.where(np.array(dis_arr) == max_dis)[-1][-1]
    max_index = np.argmax(dis_arr)
    max_frame = frame_arr[max_index]

    sort_arr = sorted(dis_arr, reverse = True)
    sort_index = np.argsort(-np.array(dis_arr))
    sort_frame_top_5 = np.array(frame_arr)[sort_index[:5]]
    avg_top_5 = np.mean(sort_arr[:5])
    avg_top_frame_5 = np.mean(sort_frame_top_5)    

    dis_frame = zip(dis_arr, frame_arr)
    dis_frame_sort = sorted(dis_frame, key = lambda x: x[0], reverse = True)
    dis_mean = np.mean(list(list(zip(dis_frame_sort))[0])[:5])
    frame_mean = np.mean(list(list(zip(dis_frame_sort))[1])[:5])


    #print(max_dis, max_frame, avg_top_5, avg_top_frame_5)
    #print(dis_mean, frame_mean)

    return max_dis, max_frame, avg_top_5, avg_top_frame_5
def MOV(mov):
    Mov=[]
    for i in mov:
        if len(i.split('.'))>1:
            if i.split('.')[1]=="avi":
                Mov.append(i)
    return Mov

if __name__ == '__main__': 
    start = time.time()
    f = open('yingxiaowei1.csv', 'w',newline='')
    writer = csv.writer(f)
    writer.writerow(['Num', 'Max_Dis', 'Max_Frame', 'Avg_Dis_Top_5', 'Avg_Frame_Top_5', 'Color_mean_velocity']) 
    path = 'D://python/video'
    hy_video_list = os.listdir(path)
    hy_video_list=MOV(hy_video_list)
    Record=[]
    #hy_video_list = ['MVI_1992.MOV']
    #hy_video_list = ['hy_211.avi']
    for video in hy_video_list:
        print('%s Start'%(video))
        optical_flows, color_velocity ,hy= features(os.path.join(path, video))
        print('%s is complated'%(video))
        record = [video[:-3]+'avi']
        record.extend([i for i in list(optical_flows)+[color_velocity]])
        print(record)
        Record.append(record)
    writer.writerows(Record)
    f.close()
    print(time.time()-start)