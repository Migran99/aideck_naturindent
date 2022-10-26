import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor
import utils
import argparse
from copy import copy



def main():

    parser = argparse.ArgumentParser(description='find signature')
    parser.add_argument('-v', '--video', required=False, help='video name in folder pallet_videos')
    #parser.add_argument('-w','--webcam', required=False,default=False)
    parser.add_argument('-n', '--name', required=False, default='video' ,help='signature name')
    parser.add_argument('-r', '--resize', required=False, default=1.0, type=float ,help='reseize factor')
    parser.add_argument('-t', '--threshold', required=False, default=0.5, type=float ,help='confidence threshold')
    args = parser.parse_args()
    

    backbone = 'pcb_p4'
    model_path = "models/model.pth.tar".format(backbone)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    extractor = FeatureExtractor(
        model_name=backbone,
        model_path=model_path,
        device=device
    )

    if args.video == None:
        cap = cv2.VideoCapture(3)
        save_name = "webcam"
        print("Webcam selected!")
    else:
        print("Video [{}] selected!".format(args.video))
        cap = cv2.VideoCapture('pallet_videos/' + args.video + '.avi')
        save_name = args.video

    save_name = args.name

    conf_th = args.threshold
    nms_th = 0.3
    factor_model=4

    net = cv2.dnn.readNet("models/pallet_block.weights", "models/pallet_block.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(factor_model*128, factor_model*128), scale=1/255, swapRB=True, crop=False)

    resize_fac=args.resize
    cam_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Capture of size: {}x{}".format(cam_height,cam_width))
    print("After resize: {}x{}".format(round(cam_height*resize_fac),round(cam_width*resize_fac)))


    # Movement filter to detect boxes
    center_frame = (int(resize_fac*cam_width/2), int(resize_fac*cam_height/2))
    max_dist_left = round(cam_width*resize_fac/6.5)
    min_dist_right = round(cam_width*resize_fac/6.0)

    # Blocks detection and identification
    pallet = []
    next_pallet_block=True
    pallet_block_idx=0
    signature_img_size = (128, 384)
    #signature_img_size = (384, 128)

    while (True):
        ret, frame = cap.read()
        if np.shape(frame) != () and not len(pallet)==3:
            # resize for faster detection/inference time
            resized = utils.resize_frame(frame, resize_fac)
            # visualization of optimal detection area
            resized = utils.draw_scan_area(resized)
            # detect box
            box, resized = utils.detect_box(resized, model, conf_th, nms_th)
            if box is not None: # and len(pallet)!=3:
                # get center from detected box and calc horizontal/x distance
                center_box = (int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2)))
                dist_x = center_frame[0] - center_box[0]
                #print(dist_x)
                cv2.putText(resized, str(pallet_block_idx+1), (box[0]+int(box[2]/2.75), box[1]+int(box[3]/1.75)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,200,220), 7)
                if dist_x <= max_dist_left and dist_x > 0: #  if pallet_block is left from center of camera and nearer to center
                    #print(len(pallet))
                    if len(pallet) < 3: # if pallet not complete
                        box_resized = (box / resize_fac).astype(int)
                        cropped_img = frame[box_resized[1]:box_resized[1]+box_resized[3], box_resized[0]:box_resized[0]+box_resized[2]]
                        cropped_img = cv2.resize(cropped_img, signature_img_size)
                        # print("pbidx: " + str(pallet_block_idx))
                        
                        if next_pallet_block: # add new pb
                            pallet.append(cropped_img)
                            next_pallet_block=False
                        else: # Update current pb
                            pallet[pallet_block_idx]=cropped_img
                if dist_x < 0 and not next_pallet_block and abs(dist_x) > min_dist_right : # Trigger for next pb. To the right and with a distance threshold
                    next_pallet_block=True
                    pallet_block_idx=pallet_block_idx+1
            cv2.imshow("Frame", resized)
        elif(len(pallet)==3):
            for id,i in enumerate(pallet):
                cv2.imwrite('tests/'+ save_name + "-" + str(id) + ".png",i)
            file1 = open('tests/'+ save_name + ".txt", "a")  # write mode
            print("Pallet complete")
            # here signatur-creation from pallet[2]=pb1 pallet[1]=pb2 pallet[0]=pb3
            pallet_vectors = utils.create_vector(extractor, pallet)
            #print(pallet_vectors[0])
            #print(pallet_vectors[1])
            #print(pallet_vectors[2])
            cv2.putText(pallet[2], "1", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)
            cv2.putText(pallet[1], "2", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)
            cv2.putText(pallet[0], "3", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)

            #cv2.putText(pallet[2], "1", (110,20), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)
            #cv2.putText(pallet[1], "2", (110,20), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)
            #cv2.putText(pallet[0], "3", (110,20), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0,200,220), 8)


            im_v_resize=utils.hconcat_resize_min([pallet[2], pallet[1], pallet[0]])
            cv2.putText(im_v_resize, "q=quit", (20,310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,220), 3)
            #im_v_resize=utils.resize_frame(im_v_resize, 0.65)
            pallet_id, min_dist = utils.find_pallet_from_db(pallet_vectors, file1)
            print("\nResult: pallet_id " + pallet_id + " min_dist " + str(min_dist) + "\n") 
            file1.write("\nResult: pallet_id " + pallet_id + " min_dist " + str(min_dist) + "\n")
            file1.close()             
            cv2.putText(im_v_resize, pallet_id, (20,210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,220), 4)
            while (k!=ord('q')):
                k=cv2.waitKey(500)
                cv2.imshow("pallet_feets",  im_v_resize)
                if k==ord('q'):
                    return
        k=cv2.waitKey(1)
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()