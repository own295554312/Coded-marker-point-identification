# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:16:01 2019

@author: 23872
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:40:29 2019

@author: 34086
"""
import sys
from progress.bar import ShadyBar
import argparse
import numpy as np
from numpy import *
import cv2
import os
import matplotlib.pyplot as plt
import math
import time
from PIL import Image
from PIL import ImageDraw
from DrawCCT import *
from Support import *

path1=os.path.abspath('.') #当前文件夹所处的绝对路径

def  com_Focus(box):
    a = max(box[1][0],box[1][1])/2
    b = min(box[1][0],box[1][1])/2
    c = math.sqrt(a**2-b**2)
    p_1_x = box[0][0] - c * math.cos(math.radians(box[2]))
    p_1_y = box[0][1] - c * math.sin(math.radians(box[2]))
    p_2_x = box[0][0] + c * math.cos(math.radians(box[2]))
    p_2_y = box[0][1] + c * math.sin(math.radians(box[2]))
    return c, p_1_x,p_1_y,p_2_x,p_2_y

def getImg(img_path):
    [dirname,filename]=os.path.split(img_path)        
    filename_str=filename.split('.')
#    print(filename_str)
    if filename_str[1]=='txt' or filename_str[1]=='xml':
        return
    if filename_str[1]=='jpg' or filename_str[1]=='JPG' or filename_str[1]=='png' or filename_str[1]=='bmp':         
        #读入图像，我日妈还得用自己的函数读图片，或者路径中不要出现英文也可以
        img=cv_imread(img_path)
        return img
    
def CCT_extract(img,N,R,color='white'):

    #存放解码结果的list
    CodeTable=[]
    '''
    image.shape[0], 图片垂直尺寸
    image.shape[1], 图片水平尺寸
    image.shape[2], 图片通道数
    '''
    img_shape=img.shape
    img_height=img_shape[0]
    img_width=img_shape[1]
    
#    print('img_width=',img_width)
#    print('img_height=',img_height)

    #将输入图像转换为灰度图
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    #使用Otsu对图像进行自适应二值化
    retval,img_bina=cv2.threshold(img_gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #使用findcontours函数对二值化后的图像进行轮廓提取,第三个参数为轮廓点的存储方式，这里选返回所有轮廓点，方便后面做筛选
    contours, hierarchy = cv2.findContours(img_bina,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)  
    #cv2.drawContours(img,contours,-1,(0,0,255),1) 
    #遍历提取出来的轮廓，筛选其中的椭圆轮廓    
    for contour in contours:
        area=cv2.contourArea(contour,False)
        length=cv2.arcLength(contour,True)
        #计算轮廓的圆度
        R0=2*math.sqrt(math.pi*area)/(length+1)
        if R0<R:                
            continue
        if len(contour)<40:
            continue
        #在原图上绘制该条轮廓

        # cv2.drawContours(img,contour,-1,(0,0,255),1)

        # plt.imshow(img)
        # plt.show()

#            print('det_r=',det_r)
#            print('len(contour)=',len(contour))
        #将轮廓点集转换为numpy数组
        e_points=np.array(contour) 
        #得到拟合的椭圆参数：中心点坐标，尺寸，旋转角
        box1=cv2.fitEllipse(e_points)

        #print('box1:',box1)
        box2=tuple([box1[0],tuple([box1[1][0]*2,box1[1][1]*2]),box1[2]])
        box3=tuple([box1[0],tuple([box1[1][0]*3,box1[1][1]*3]),box1[2]])

        cl, pl_1_x, pl_1_y, pl_2_x, pl_2_y = com_Focus(box1)
        cb, pb_1_x, pb_1_y, pb_2_x, pb_2_y = com_Focus(box2)


        #求得最外层椭圆的最小外接矩形的四个顶点，顺时针方向
        minRect = cv2.boxPoints(box3)     
        #计算椭圆的长轴
        a=max(box3[1][0],box3[1][1])
        #长轴的修正值
        s=a+10
        #在原图像中裁剪CCT所在的区域
        cct_roi=None
        row_min=round(box1[0][1]-s/2)
        row_max=round(box1[0][1]+s/2)
        col_min=round(box1[0][0]-s/2)
        col_max=round(box1[0][0]+s/2)


        a_2 = max(box2[1][0], box2[1][1])
        s_2 = a_2
        row_2_min = round(box1[0][1] - s_2 / 2)
        row_2_max = round(box1[0][1] + s_2 / 2)
        col_2_min = round(box1[0][0] - s_2 / 2)
        col_2_max = round(box1[0][0] + s_2 / 2)

        #print('判断该ROI是否超出边界......')
        #print([row_min,row_max,col_min,col_max])
        #判断cct_roi是否超出原图像边界
        if row_min>=0 and row_max<=img_height and col_min>=0 and col_max<=img_width:
            #从原图像中将cct_roi截取出来            
            cct_roi=img[row_min:row_max,col_min:col_max]

            # plt.imshow(cct_roi)
            # plt.show()

            #cct_roi相对于原始影像的偏移量
            dx=box1[0][0]-s/2
            dy=box1[0][1]-s/2
            #对CCT椭圆区域进行仿射变换将其变为正圆
            src=np.float32([[minRect[0][0]-dx,minRect[0][1]-dy],[minRect[1][0]-dx,minRect[1][1]-dy],
                            [minRect[2][0]-dx,minRect[2][1]-dy],[minRect[3][0]-dx,minRect[3][1]-dy],
                            [box1[0][0]-dx,box1[0][1]-dy]])
            dst=np.float32([[box1[0][0]-s/2-dx,box1[0][1]-s/2-dy],[box1[0][0]+s/2-dx,box1[0][1]-s/2-dy],
                            [box1[0][0]+s/2-dx,box1[0][1]+s/2-dy],[box1[0][0]-s/2-dx,box1[0][1]+s/2-dy],
                            [box1[0][0]-dx,box1[0][1]-dy]])        
            #得到仿射变换矩阵
            #M=cv2.getAffineTransform(src,dst)
            M=my_getAffineTransform(src,dst)
            M[0][2] = M[0][2] + 10
            M[1][2] = M[1][2] + 10
            if isinstance(M,int):
                continue
            #计算仿射变换后的中心点坐标
            X0,Y0=PointAffineTransform(M,[box1[0][0]-dx,box1[0][1]-dy])
            #print('X0=',X0,'  ','Y0=',Y0)
            CCT_img=None
            #对cct_roi进行仿射变换
            cct_roi_size=np.shape(cct_roi)
            if cct_roi_size[0]>0 and cct_roi_size[1]>0:
                CCT_img=cv2.warpAffine(cct_roi,M,(round(s)+20,round(s)+20))

            # plt.imshow(CCT_img)
            # plt.show()

            #print('cct img shape=',np.shape(CCT_img))
            #对仿射变换后的CCT进行缩放
            #CCT_large = cv2.resize(CCT_img, (0, 0), fx=200.0/s, fy=480.0/s, interpolation=cv2.INTER_LANCZOS4)
            #将放大后的CCT转换为灰度图
            #CCT_gray=cv2.cvtColor(CCT_large,cv2.COLOR_BGR2GRAY)
            CCT_gray = cv2.cvtColor(CCT_img, cv2.COLOR_BGR2GRAY)

            # plt.imshow(CCT_gray)
            # plt.show()

#           #对该灰度图进行自适应二值化
            retval,CCT_bina=cv2.threshold(CCT_gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#           #执行腐蚀
            CCT_eroded=cv2.erode(CCT_bina,kernel)
            #
            # plt.imshow(CCT_bina)
            # plt.show()

            # plt.imshow(CCT_eroded)
            # plt.show





            listx = []
            listy = []

            list_l_x = []
            list_l_y = []

            for k in range(row_2_min,row_2_max ):
                for j in range(col_2_min , col_2_max ):

                    drel_1 = math.sqrt((j - pl_1_x)**2 + (k - pl_1_y)**2)
                    drel_2 = math.sqrt((j - pl_2_x)**2 + (k - pl_2_y)**2)
                    dreb_1 = math.sqrt((j - pb_1_x)**2 + (k - pb_1_y)**2)
                    dreb_2 = math.sqrt((j - pb_2_x)**2 + (k - pb_2_y)**2)

                    if (drel_1 + drel_2) > max(box1[1][0],box1[1][1]) and (dreb_1 + dreb_2) < max(box2[1][0],box2[1][1]):
                        listy.append(j)
                        listx.append(k)


                    if((drel_1 + drel_2) < max(box1[1][0],box1[1][1])):
                        list_l_y.append(j)
                        list_l_x.append(k)

            draw = np.zeros((shape(img_gray)[0],shape(img_gray)[1]))
            for i in range(0,len(listx)):
                draw[listx[i],listy[i] ] = 255

            # plt.subplot(1, 2, 1)
            # plt.imshow(draw)
            # plt.subplot(1, 2, 2)
            # plt.imshow(img)
            # plt.show()

            draw_l = np.zeros((shape(img_gray)[0],shape(img_gray)[1] ))
            for i in range(0,len(list_l_x)):
                draw_l[list_l_x[i],list_l_y[i] ] = 255

            # plt.subplot(1, 2, 1)
            # plt.imshow(draw_l)
            # plt.subplot(1, 2, 2)
            # plt.imshow(img)
            # plt.show()


            lens = len(listx)
            arr = np.zeros((lens,1))

            for i in range(0,lens):
                arr[i] = img_gray[listx[i],listy[i]]
            arr_mean = np.mean(arr)
            arr_var = np.var(arr)
            arr_mean_list.append(arr_mean)
            arr_var_list.append(arr_var)

            lens_l = len(list_l_x)
            arr_l = np.zeros((lens_l,1))

            for t in range(0,lens_l):
                arr_l[t] = img_gray[list_l_x[t],list_l_y[t]]
            arr_l_mean = np.mean(arr_l)
            arr_l_var = np.var(arr_l)
            arr_l_mean_list.append(arr_mean)
            arr_l_var_list.append(arr_var)

            ratio_mean = arr_l_mean / arr_mean
            ratio_var  = arr_l_var  / arr_var


            #if CCT_or_not(CCT_eroded):
            ToF,rrrr = cct_or_not(CCT_bina)

            if ToF==1:

           # if cct_or_not(CCT_bina):




#                print('len(contour)=',len(contour))
#                print('a=',box3[1][0]/3)
#                print('b=',box3[1][1]/3)
#                print('s=',s)
#                print('R0=',R0)
                #调用解码函数进行解码
                code=CCT_Decode(CCT_eroded,N,rrrr,color)
                #
                # CodeTable.append([code, box1[0][0], box1[0][1]])
                # # print([code,box1[0][0],box1[0][1]])
                # # 将编码在原图像中绘制出来.各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                # cv2.putText(img, str(code), (int(box3[0][0] - 0.25 * s), int(box1[0][1] + 0.5 * s)), cv2.FONT_HERSHEY_COMPLEX, 1,
                #             (0, 0, 255), 2)
                # # 绘制拟合出的椭圆
                # cv2.ellipse(img, box1, (0, 255, 0), 1)
                # cv2.ellipse(img, box2, (0, 255, 0), 1)
                # cv2.ellipse(img, box3, (0, 255, 0), 1)

                if ratio_mean > 2.5:

                        CodeTable.append([code,box1[0][0],box1[0][1]])
                            #print([code,box1[0][0],box1[0][1]])
                            #将编码在原图像中绘制出来.各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                        cv2.putText(img,str(code),(int(box3[0][0]-0.25*s),int(box1[0][1]+0.5*s)),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            #绘制拟合出的椭圆
                        cv2.ellipse(img,box1,(0,255,0),1)
                        cv2.ellipse(img,box2,(0,255,0),1)
                        cv2.ellipse(img,box3,(0,255,0),1)
    return CodeTable,img

#判断给定图像中是否有CCT
# def CCT_or_not(cct_bimg):
# #    plt.imshow(cct_bimg)
# #    plt.show()
# #    print("正在判断该区域是否包含CCT......")
#
#
#
#     sample_num=36
#     #得到cct图像的尺寸
#     shape=np.shape(cct_bimg)
#     height=shape[0]
#     width=shape[1]
#     #得到中心点坐标，该坐标可以视为CCT中心坐标
#     X0=width/2
#     Y0=height/2
#     #得到CCT中心⚪的半径
#     r1=X0/3.0
#     #存放三个圆周上采样点像素值的列表
#     pixels_0_5_r=[]
#     #pixels_0_7_5_r=[]
#     pixels_1_5_r=[]
#     pixels_2_5_r=[]
# #    pixels_3_5_r=[]
#     #遍历三个圆周处的采样点像素
#     for j in range(sample_num):
#         xi=math.cos(10.0*j/180*math.pi)
#         yi=math.sin(10.0*j/180*math.pi)
#
#         x_0_5_r=0.5*r1*xi+X0
#         y_0_5_r=0.5*r1*yi+Y0
#
# #        x_0_7_5_r=0.85*r1*xi+X0
# #        y_0_7_5_r=0.85*r1*yi+Y0
#
#         x_1_5_r=1.5*r1*xi+X0
#         y_1_5_r=1.5*r1*yi+Y0
#
#         x_2_5_r=2.5*r1*xi+X0
#         y_2_5_r=2.5*r1*yi+Y0
#
# #        x_3_5_r=3.5*r1*xi+X0
# #        y_3_5_r=3.5*r1*yi+Y0
#         #妈卖批！！为什么访问像素要Y在前面啊！！！卡了老子两个小时!
#         #好吧，我认输，以后访问像素记住【rows,cols】，先行后列
#         pixel_value_0_5_r=cct_bimg[round(y_0_5_r)][round(x_0_5_r)]
# #       pixel_value_0_7_5_r=cct_bimg[round(y_0_7_5_r)][round(x_0_7_5_r)]
#         pixel_value_1_5_r=cct_bimg[round(y_1_5_r)][round(x_1_5_r)]
#         pixel_value_2_5_r=cct_bimg[round(y_2_5_r)][round(x_2_5_r)]
# #       pixel_value_3_5_r=cct_bimg[round(y_3_5_r)][round(x_3_5_r)]
#
#         pixels_0_5_r.append(pixel_value_0_5_r)
# #       pixels_0_7_5_r.append(pixel_value_0_7_5_r)
#         pixels_1_5_r.append(pixel_value_1_5_r)
#         pixels_2_5_r.append(pixel_value_2_5_r)
# #        pixels_3_5_r.append(pixel_value_3_5_r)
# #    print(pixels_0_7_5_r)
# #    print(sum(pixels_1_5_r)/sample_num*1.0)
# #    print(sum(pixels_2_5_r)/sample_num*1.0)
# #    print(sum(pixels_3_5_r)/sample_num*1.0)
#
#     if sum(pixels_0_5_r)==sample_num and sum(pixels_1_5_r)==0 and sum(pixels_2_5_r)>=2:
# #        print('判断完成：True')
#         return True
#     elif sum(pixels_0_5_r)==0 and sum(pixels_1_5_r)==sample_num and sum(pixels_2_5_r)<=sample_num-2:
#         return True
#     else:
# #        print('判断完成：False')
#         return False

def cct_or_not (test_img):
    s = False
    rrr = 0

    contours_CCT, hierarchy_CCT = cv2.findContours(test_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # 遍历提取出来的轮廓，筛选其中的椭圆轮廓
    for contour_CCT in contours_CCT:
        area_CCT = cv2.contourArea(contour_CCT, False)
        length_CCT = cv2.arcLength(contour_CCT, True)
        # 计算轮廓的圆度
        R0 = 2 * math.sqrt(math.pi * area_CCT) / (length_CCT + 1)
        if R0 < 0.90:
            continue
        if len(contour_CCT) < 40:
            continue
        # 在原图上绘制该条轮廓

        # cv2.drawContours(CCT_bina,contour_CCT,-1,(0,0,255),10)
        #
        # plt.imshow(CCT_bina)
        # plt.show()

        # 将轮廓点集转换为numpy数组
        e_points_CCT = np.array(contour_CCT)
        # 得到拟合的椭圆参数：中心点坐标，尺寸，旋转角
        box1_CCT = cv2.fitEllipse(e_points_CCT)

        print()

        shape=np.shape(test_img)
        height=shape[0]
        width=shape[1]
        #得到中心点坐标，该坐标可以视为CCT中心坐标
        X0=width/2
        Y0=height/2
        #得到CCT中心⚪的半径
        rrr= int(min(box1_CCT[1][0],box1_CCT[1][1])/2)

        sample_num = 36
        pixel_num_mean = 0
        find_num = 0
        pixel_num = 0

        for i in range(sample_num):
            xi = math.cos(10 * i / 180 * math.pi)
            yi = math.sin(10 * i / 180 * math.pi)

            x_2_5_r = 2.5 * rrr * xi + X0
            y_2_5_r = 2.5 * rrr * yi + Y0

            if 0 < x_2_5_r < width-10 and 0 < y_2_5_r < height-10:
                if test_img[round(y_2_5_r)][round(x_2_5_r)] == 1:
                    find_num = find_num + 1
                    for t in range(2 * rrr, int(X0)):
                        x_t = t * xi + X0
                        y_t = t * yi + Y0
                        if 0 <  x_t < width  and 0 < y_t < height:
                            if test_img[round(y_t)][round(x_t)] == 1:
                                pixel_num = pixel_num + 1
                    print()
        if find_num != 0:
           pixel_num_mean = pixel_num / find_num
        print()
        print()

        if pixel_num_mean > rrr*0.85 and pixel_num_mean < rrr * 1.15:
           s=True
           break
        else:
           s=False
           break
    return s,rrr

def CCT_Decode(cct_img,N,rrrrr,color):
    # plt.imshow(cct_img)
    # plt.show()
    #得到cct图像的尺寸
    shape=np.shape(cct_img)
    height=shape[0] 
    width=shape[1]
    #得到中心点坐标，该坐标可以视为CCT中心坐标
    X0=width/2
    Y0=height/2
    #得到CCT的三个半径
    r1=rrrrr
    #存放所有码值的list
    code_all=[]
    #如果是12位编码，那就转30圈，每圈加一度
    for j in range(int(360/N)):
        code_j=[]#存放单圈码值的list
        #以N为标准在2.5倍r1的圆环上均匀采样
        for k in range(N):
            x=2.5*r1*math.cos((360.0/N*k+j)/180*math.pi)+X0
            y=2.5*r1*math.sin((360.0/N*k+j)/180*math.pi)+Y0
            #访问像素记住【rows,cols】，先行后列
            pixel_value=cct_img[round(y)][round(x)]
            code_j.append(pixel_value)
        #将每一圈得到的编码都转换为最小编码
        temp1=B2I(code_j,N)
        temp2=I2B(temp1,N)
        code_all.append(temp2) 
    code=np.asarray(code_all)
#    print(code)
    #对列求平均
    code=np.mean(code,0)    
#    print(code)
    #对cct编码进行二值化
    result=[]         
    for i in code:
        if i>0.5:
            result.append(1)
        if i<=0.5:
            result.append(0)
    if color=='black':
        result=swap0and1(result)
    #调用DrawCCT中的解码函数进行解码
    return B2I(result,len(result))

#反转编码的函数：0->1,1->0,为黑色码带设计
def swap0and1(code_list):
    result=[]
    for i in code_list:
        if i==0:
            result.append(1)
        if i==1:
            result.append(0)
    return result

#循环遍历data文件
def scanne_data_dir(src_folder,save_folder,N,R,color):
    file_list=os.listdir(src_folder)
    file_num=len(file_list)    
    bar = ShadyBar('Processing', max=file_num)
    for root,dirs,files in os.walk(src_folder):
        for file in files:
            bar.next()
            name_str=file.split('.')
            if name_str[1]=='JPG' or name_str[1]=='jpg' or name_str[1]=='png' or name_str[1]=='PNG' or name_str[1]=='bmp':
                path=os.path.join(root,file)
                img=getImg(path)
                code,result_img=CCT_extract(img,N,R,color)
                cv2.imwrite(save_folder+file,img)
    bar.finish()           
#分割命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser(description='Detect and Decode the CCT from image.')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='images path',
                        default='D:\\a1\\', type=str)
    parser.add_argument('--filename', dest='filename',
                        help='image filename',
                        default='ture01.jpg', type=str)
    parser.add_argument('--bit_n', dest='bit_n',
                        help='CCT bit number',
                        default=12, type=int)
    parser.add_argument('--threshold', dest='threshold',
                        help='threshold for contour filter, which between 0 and 1',
                        default=0.85, type=float)
    parser.add_argument('--save_folder', dest='save_folder',
                        help='the folder for saving the processed image',
                        default='D:\\result_a1\\', type=str)
    parser.add_argument('--batch', dest='batch',
                        help='batch processing or not',
                        default=False, type=bool)
    parser.add_argument('--color', dest='color',
                        help='the color of CCT mark(white over black / black over white)',
                        default='white', type=str)
    
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args
    
#调用程序
if __name__ == '__main__':  
   t0 = time.perf_counter()
   args = parse_args()

   arr_mean_list = []
   arr_var_list = []

   arr_l_mean_list = []
   arr_l_var_list = []

   print('Called with args:')
   print(args)
   if not args.batch:
       image_path=os.path.join(args.src_folder,args.filename)
       print('image path: ',image_path)

       raw_img = cv2.imread(image_path)
       #raw_img=getImg(image_path)

       code_table,processed_img=CCT_extract(raw_img,args.bit_n,args.threshold,args.color)
       save_path=os.path.join(args.save_folder,args.filename)
       cv2.imwrite(save_path,processed_img)
   #print(code_table)
   if args.batch:
       scanne_data_dir(args.src_folder,args.save_folder,args.bit_n,args.threshold,args.color)
   t1 = time.perf_counter()
   print('\n process time:',t1-t0)
   print(arr_var_list)
   print(arr_mean_list)
    # raw_img=getImg('./data/cct8_black.png')
    # code_table,processed_img=CCT_extract(raw_img,8,0.85,'black')
    # cv2.imwrite('./result/cct8_black.png',processed_img)
#    print(code_table)
 #   scanne_data_dir('./100_0004/','./result/',12,0.8)
