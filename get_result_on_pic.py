#coding=utf-8
import cv2
import json
import random
result_json = './results/CornerNet/50000/testing/res1.json'
detections = []
with open(result_json, "r", encoding='utf-8') as f:
    detections = json.load( f)


im_name = detections[1]['image_id']
imfile = './data/coco/images/testdev2017/' + '000000' + '{}'.format(im_name) + '.jpg'
img = cv2.imread(imfile)
for i in range(0, len(detections)):
    c = random.randint(100,200)
    cls_name = detections[i]['category_id']
    confi = detections[i]['score']
    if confi < .4:
        continue
    box = detections[i]['bbox']
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (int(c * 0.1), int(c * 0.5), c), 1)
    # cv2.putText(img, '{}'.format(cls_name) + '  ' + '{:.3f}'.format(confi), (int(box[0]), int(box[1] - 10)),
    #             cv2.FONT_ITALIC, 1, (int(c * 0.1), int(c * 0.5), c), 1)
    while (box):
        box.pop(-1)
cv2.imshow('Detecting image...', img)
# timer.total_time = 0
cv2.waitKey(0)