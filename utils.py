import cv2
import pybase64 as base64
import numpy as np
import ctypes
from scipy import spatial
import os
from torchreid import utils

def get_next_pallet_id():
    allfiles=os.listdir('pallet_signatures')
    if len(allfiles) == 0:
        return 1
    else:
        list_ids_int=[x.split('.')[0] for x in allfiles]
        sorted_list_ids = sorted([ int(x) for x in list_ids_int ])
        return sorted_list_ids[len(sorted_list_ids)-1] + 1

def save_pallet_foot(pallet_vector):
    new_pallet_id = get_next_pallet_id()
    with open('pallet_signatures/' + str(new_pallet_id) + '.npy', 'wb') as f:
        np.save(f, pallet_vector)


def create_vector(model, pallet, interpolation=cv2.INTER_CUBIC):
    vector = []
    for i in range(len(pallet)):
        resized = cv2.resize(pallet[i], (128, 384), interpolation=interpolation)
        vector.append(model(resized).cpu().numpy())
    return vector


def save_pallet_input_id(pallet, pallet_name):
    #new_pallet_id = input('Geben Sie die gewünschte Paletten-ID ein:')
    with open('pallet_signatures/' + str(pallet_name) + '.npy', 'wb') as f:
        np.save(f, pallet[0])
        np.save(f, pallet[1])
        np.save(f, pallet[2])

def get_weight(value):
    val_round = round(value,2)
    if val_round<=0.01:
        return 10
    elif val_round>0.01 and val_round < 0.1:
        return 8
    elif val_round>=0.7:
        return 1
    else:
        val_int = int(round(val_round,2)*10)
        weights = {1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1}
        return weights.get(val_int)

def save_pallet_to_db(pallet):
    new_pallet_id = get_next_pallet_id()
    with open('pallet_signatures/' + str(new_pallet_id) + ".npy", 'wb') as f:
        np.save(f, pallet[0])
        np.save(f, pallet[1])
        np.save(f, pallet[2])

def compare_pallets(pallet1, pallet2,file): # euclidean or cosine
    dist_pb1 = round(spatial.distance.cdist(pallet1[0], pallet2[0], 'euclidean')[0][0],6)
    dist_pb2 = round(spatial.distance.cdist(pallet1[1], pallet2[1], 'euclidean')[0][0],6)
    dist_pb3 = round(spatial.distance.cdist(pallet1[2], pallet2[2], 'euclidean')[0][0],6)
    dist_pb1_weight = get_weight(dist_pb1)
    dist_pb2_weight = get_weight(dist_pb2)
    dist_pb3_weight = get_weight(dist_pb3)
    #mean = (dist_pb1 + dist_pb2 + dist_pb3) / 3 
    weighted_mean = (dist_pb1*dist_pb1_weight + dist_pb2*dist_pb2_weight + dist_pb3*dist_pb3_weight) / (dist_pb1_weight + dist_pb2_weight + dist_pb3_weight)
    print(str(round(dist_pb1, 4)) + "/" + str(dist_pb1_weight) + " " + str(round(dist_pb2,4)) + "/" + str(dist_pb2_weight) + " " + str(round(dist_pb3,4)) + "/" + str(dist_pb3_weight))
    print(weighted_mean)
    print("--------------------------")
    file.write("\n"+str(round(dist_pb1, 4)) + "/" + str(dist_pb1_weight) + " " + str(round(dist_pb2,4)) + "/" + str(dist_pb2_weight) + " " + str(round(dist_pb3,4)) + "/" + str(dist_pb3_weight))
    file.write("\n"+str(weighted_mean))
    return round(weighted_mean, 5)

def find_pallet_from_db(pallet, file):
    min_dist=2.0
    pallet_id='empty'
    for filename in os.listdir('pallet_signatures'):
        if filename.endswith('.npy'): 
            path = os.path.join('pallet_signatures', filename)
            print(path)
            file.write("\n"+str(path))
            pallet_vector = []
            with open(path, 'rb') as f:
                pallet_vector.append(np.load(f))
                pallet_vector.append(np.load(f))
                pallet_vector.append(np.load(f))
            current_dist = compare_pallets(pallet, pallet_vector,file)
            if min_dist > current_dist:
                min_dist = current_dist
                pallet_id = filename.replace('.npy', "")
            continue
        else:
            continue
    return pallet_id, min_dist

def init_dict(waren_dictionary):
    for filename in os.listdir('pallet_signatures'):
        if filename.endswith('.npy'): 
            pallet_id = filename.replace('.npy', "")
            waren_dictionary.pop(pallet_id)
    return waren_dictionary

def encode2base64(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    # ret = False
    if ret == True:
        return base64.b64encode(buffer)
    else:
        return None

def resize_frame(frame, resize_factor=1.0):
    width = int(frame.shape[1] * resize_factor)
    height = int(frame.shape[0] * resize_factor)
    dim = (width, height)
    img_original=cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
    return img_original

def draw_filled_rect(img_original, box, label, alpha=0.05):
    color = [10, 220, 10]
    img_rect_filled = img_original.copy()
    cv2.rectangle(img_rect_filled, box , color=(0, 225, 0), thickness=-1)
    img_processed=cv2.addWeighted(img_rect_filled, alpha, img_original, 1 - alpha, 0)
    img_processed=cv2.rectangle(img_processed, box , color=(0, 225, 0), thickness=3)
    cv2.putText(img_processed, label + " %", (380, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return img_processed

def draw_scan_area(resized):
    color = [40, 220, 50]
    left_dist=0.1
    right_dist=0.9
    top_dist=0.1
    bot_dist=0.85
    length = 0.15
    thickn=4
    pt_tl=(int(resized.shape[1]*left_dist), int(resized.shape[0]*top_dist))
    pt_tl_r=(int(resized.shape[1]*(0.1+length)), int(resized.shape[0]*top_dist))
    pt_tl_b=(int(resized.shape[1]*0.1), int(resized.shape[0]*(top_dist+length)))
    pt_tr=(int(resized.shape[1]*right_dist), int(resized.shape[0]*top_dist))
    pt_tr_l=(int(resized.shape[1]*(right_dist-length)), int(resized.shape[0]*top_dist))
    pt_tr_b=(int(resized.shape[1]*right_dist), int(resized.shape[0]*(top_dist+length)))
    pt_bl=(int(resized.shape[1]*left_dist), int(resized.shape[0]*bot_dist))
    pt_bl_r=(int(resized.shape[1]*(left_dist+length)), int(resized.shape[0]*bot_dist))
    pt_bl_t=(int(resized.shape[1]*left_dist), int(resized.shape[0]*(bot_dist-length)))
    pt_br=(int(resized.shape[1]*right_dist), int(resized.shape[0]*bot_dist))
    pt_br_l_=(int(resized.shape[1]*(right_dist-length)), int(resized.shape[0]*bot_dist))
    pt_br_t=(int(resized.shape[1]*right_dist), int(resized.shape[0]*(bot_dist-length)))
    cv2.line(resized, pt_tl, pt_tl_r, color, thickn)
    cv2.line(resized, pt_tl, pt_tl_b, color, thickn)
    cv2.line(resized, pt_tr, pt_tr_l, color, thickn)
    cv2.line(resized, pt_tr, pt_tr_b, color, thickn)
    cv2.line(resized, pt_bl, pt_bl_r, color, thickn)
    cv2.line(resized, pt_bl, pt_bl_t, color, thickn)
    cv2.line(resized, pt_br, pt_br_l_, color, thickn)
    cv2.line(resized, pt_br, pt_br_t, color, thickn)
    return resized

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, ( int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def draw_filled_rect(img_original, box, label, alpha=0.05):
    color = [10, 220, 10]
    img_rect_filled = img_original.copy()
    [height,width,_] = img_original.shape
    cv2.rectangle(img_rect_filled, box , color=(0, 225, 0), thickness=-1)
    img_processed=cv2.addWeighted(img_rect_filled, alpha, img_original, 1 - alpha, 0)
    img_processed=cv2.rectangle(img_processed, box , color=(0, 225, 0), thickness=3)
    cv2.putText(img_processed, label + " %", (int(height/2),int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return img_processed

def detect_box(resized, model, conf, nms):
    class_names = ['palletblock']
    classes, scores, boxes = model.detect(resized, conf, nms)
    if len(boxes) > 0:
        for (classid, score, box) in zip(classes, scores, boxes):
            if classid == 0:
                label = "%s [%d] : %.2f" % (class_names[classid],classid, 100*score)
            else:
                label = "unknown [%d] : %.2f" % (classid, 100*score)
            resized = draw_filled_rect(resized, box, label)
            return box, resized
    else:
        return None, resized