import tensorflow as tf
import cv2
import numpy as np
import os
from numpy.linalg import norm
import predict

export_path = 'path/to/insight/face/saved/pb'

people1_path = 'path/to/single/person/image/group/folder'
people1_list = os.listdir(people1_path) 
people1_imgs = np.zeros((112,112,3))

for i, img in enumerate(people1_list):
    img_path = os.path.join(people1_path,img)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(112, 112),interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    if i==0:
        people1_imgs = img
    else:
        people1_imgs = np.concatenate((people1_imgs, img), axis = 0)

people2_imgs,box_imgs = predict.predict('path/to/img/with/several/people')
people2_imgs = np.array(people2_imgs)

def simple(A, B):
       ret = np.dot(A,B)/(norm(A)*norm(B))
       return ret

with tf.Session(graph=tf.Graph()) as sess:

    loaded = tf.saved_model.loader.load(sess, ['serve'], export_path)
    x = sess.graph.get_tensor_by_name('data:0')
    y = sess.graph.get_tensor_by_name('fc1/add_1:0')
    feature = sess.run(y, feed_dict={x: people1_imgs})
    feature2 = sess.run(y, feed_dict={x: people2_imgs})

answers = []
for i,f11 in enumerate(feature2): 
    score = 0.0
    out_num = 0
    check_pt = False
    for i2,f12 in enumerate(feature):
        if score < simple(f11, f12): 
            out_num = i2
            score = simple(f11, f12)
            check_pt = True
            
    if check_pt == True:
        print("{} vs {} = {}".format(people1_list[out_num],i,score))
        answers.append(people1_list[out_num])
cv2.imshow('detected',box_imgs)
for i, img in enumerate(answers):
    image = cv2.imread(os.path.join(people1_path,img))
    cv2.imshow('detected{}'.format(i+1),image)

cv2.waitKey(10000)
cv2.destroyAllWindows()
