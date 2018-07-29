import shutil
from handle import pic_handle
import cv2
import os
# if __name__=='__main__':
#
#     for i in range(1,27):
#         pic_name = 'C:/Users/lx/Desktop/pic/'+str(i)+'.jpg'
#         if(os.path.exists('C:/Users/lx/Desktop/handle/'+str(i)) ):
#             shutil.rmtree('C:/Users/lx/Desktop/handle/'+str(i))
#         os.mkdir('C:/Users/lx/Desktop/handle/'+str(i))
#         img = cv2.imread(pic_name)
#         pic_handle(img,i)
if __name__=='__main__':

    for i in range(1,6405):
        if(i%100==0):
            print(str(i),'finished')
        pic_name = 'C:/Users/lx/Desktop/test/test_pic/image'+str(i)+'.jpg'
        if(os.path.exists('C:/Users/lx/Desktop/handle/'+str(i)) ):
            shutil.rmtree('C:/Users/lx/Desktop/handle/'+str(i))
        os.mkdir('C:/Users/lx/Desktop/handle/'+str(i))
        img = cv2.imread(pic_name)
        pic_handle(img,i)