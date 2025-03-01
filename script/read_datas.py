#-*-coding:utf-8-*-
'''
# date:2024-08
# Author: Xian
# function: read head 3d x
包括：
1）2d投影点
2）三维 mesh rgb
'''
import os
import cv2
import numpy as np
from utils import read_obj,resize_img_keep_ratio
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    name_ = "20250301"
    path_img = "datas/{}/".format(name_)
    path_ = "datas/{}_render/".format(name_)
    path_fit = "datas/{}_fit/".format(name_)
    path_3d_render =  "datas/{}_high_render/".format(name_)
    if not os.path.exists(path_3d_render): # 如果文件夹不存在
        os.mkdir(path_3d_render) # 生成文件夹
    # 选择显示2d部分
    '''
        1: brows
        2：contour
        3: eyes
        4: forehead
        5: lips
        6: nose
    '''
    define_show_face_part = 2 # 1,2,3,4,5,6

    if define_show_face_part==1:
        contour_idx = np.load("./cfg/brows.npy",allow_pickle=True)
    elif define_show_face_part==2:
        contour_idx = np.load("./cfg/contour.npy",allow_pickle=True)
    elif define_show_face_part==3:
        contour_idx = np.load("./cfg/eyes.npy",allow_pickle=True)
    elif define_show_face_part==4:
        contour_idx = np.load("./cfg/forehead.npy",allow_pickle=True)
    elif define_show_face_part==5:
        contour_idx = np.load("./cfg/lips.npy",allow_pickle=True)
    elif define_show_face_part==6:
        contour_idx = np.load("./cfg/nose.npy",allow_pickle=True)

    contour_idx = contour_idx.item()
    # 获取特定点索引
    idx_choose = []
    for k_ in contour_idx.keys():
        for j in range(contour_idx[k_].shape[0]):
            idx_choose.append(contour_idx[k_][j])

    #-------------------------------------------------------------------------------
    for f_ in os.listdir(path_img):
        if not((".jpg" in f_) or (".png" in f_)):
            continue
        '''
        加载数据
        '''
        path_render = path_ + f_.replace(".png","_rendered.png").replace(".jpg","_rendered.png") # 三维 demo 渲染图
        path_npy = path_ + f_.replace(".jpg",".npy").replace(".png",".npy") # 2d 头部关键点
        path_new_mesh_obj = path_fit + f_.replace(".jpg","_fit_mesh.obj").replace(".png","_fit_mesh.obj") # 3d 头部 关键点
        path_camera = path_ + f_.replace(".jpg","_camera.npy").replace(".png","_camera.npy") # 虚拟相机参数
        path_im = path_img + f_ # 原图

        #-----------------------------------------
        if not os.access(path_new_mesh_obj,os.F_OK):
            continue
        if not os.access(path_camera,os.F_OK):
            continue
        if not os.access(path_npy,os.F_OK):
            continue
        if not os.access(path_render,os.F_OK):
            continue
        #----------------------------------------
        # 加载数据
        img = cv2.imread(path_im) # 图像
        img2= img.copy()
        img_2d_kpt= img.copy() # 绘制2d坐标
        pts2d = np.load(path_npy) # 2d 头部关键点
        print("pts2d shape :",pts2d.shape)
        # 绘制2d关键点
        for i in range(pts2d.shape[0]):
            if i in idx_choose:
                x,y = int(pts2d[i][0]),int(pts2d[i][1])
                cv2.circle(img_2d_kpt, (x,y), 7, (255,255,0), -1) #绘制实心圆
                cv2.circle(img_2d_kpt, (x,y), 2, (0,0,255), -1) #绘制实心圆

        cv2.namedWindow("img_2d_kpt",0)
        cv2.imshow("img_2d_kpt",img_2d_kpt)
        cv2.waitKey(1)
        #---------------------------
        Mesh3D = read_obj(path_new_mesh_obj) # 3d 头部 关键点
        camera_param = np.load(path_camera) # 虚拟相机参数
        [fx,fy,cx,cy,k1,k2,k3,p1,p2,_,_,_]= list(camera_param)
        print("相机参数：fx,fy,cx,cy,k1,k2,k3,p1,p2:",fx,fy,cx,cy,k1,k2,k3,p1,p2)
        # 进行 3D 贴图渲染
        faces = Mesh3D["faces_index"] # 获取 头部mesh三角网格索引
        verts = Mesh3D["joints"]

        list_faces = []
        for ff_ in list(faces): # mesh 三角片面
            list_faces.append((ff_[2]-1,ff_[1]-1,ff_[0]-1)) # open3d 索引 从 0 开始
        #-------------------------------------------------------------------
        list_verts = []
        for vv_ in list(verts):# mesh 网格点
            list_verts.append((vv_[0],-vv_[1],vv_[2]))

        #---------------------
        m=o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(list_verts),
                                    o3d.open3d.utility.Vector3iVector(list_faces))
        m.compute_vertex_normals()
        #-----------------------------------------------------------------------
        h_,w_ = img.shape[:2]

        uv_list = []
        for ii in range(faces.shape[0]):
            c,b,a = faces[ii] # 三角网格索引

            a -=1
            b -=1
            c -=1
            # 计算向量可见性
            x1_,y1_ = int(pts2d[a][0]),int(pts2d[a][1])
            x2_,y2_ = int(pts2d[b][0]),int(pts2d[b][1])
            x3_,y3_ = int(pts2d[c][0]),int(pts2d[c][1])
            #----------------------------
            x1_,y1_ = x1_/w_,y1_/h_
            x2_,y2_ = x2_/w_,y2_/h_
            x3_,y3_ = x3_/w_,y3_/h_

            uv_list.append([x1_,y1_])
            uv_list.append([x2_,y2_])
            uv_list.append([x3_,y3_])

        v_uv = np.array(uv_list)
        m.triangle_uvs = o3d.open3d.utility.Vector2dVector(v_uv)
        m.triangle_material_ids = o3d.utility.IntVector([0]*len(faces))
        m.textures = [o3d.geometry.Image(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))]

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Head3D', width=960, height=960,visible = False)
        renderOptions = vis.get_render_option()# 获取渲染选项
        renderOptions.load_from_json('./open3d_config/render_option.json') # open3d 引擎初始化
        renderOptions.mesh_show_back_face = False # 设置不显示背面
        #
        vis.add_geometry(m)
        #
        vis.poll_events()
        vis.update_renderer()

        img_o3d = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        del renderOptions
        del vis
        del m

        # 将图像转换为 NumPy 数组
        img_o3d = np.asarray(img_o3d)
        img_o3d = (img_o3d * 255).astype(np.uint8)

        # 转换为 BGR 格式以便 OpenCV 显示
        img_o3d = cv2.cvtColor(img_o3d, cv2.COLOR_RGB2BGR)

        img = resize_img_keep_ratio(img, [img_o3d.shape[1],img_o3d.shape[0]])
        img_m = np.hstack((img,img_o3d))

        cv2.imwrite(path_3d_render + f_,img_m)
        cv2.putText(img_m, ' High precision & Head 3D X ', (540+25,60),cv2.FONT_HERSHEY_COMPLEX, 1.35, (55, 128, 55),7)
        cv2.putText(img_m, ' High precision & Head 3D X', (540+25,60),cv2.FONT_HERSHEY_COMPLEX, 1.35, (55, 158, 255),2)

        cv2.namedWindow("img_m",0)
        cv2.imshow("img_m",img_m)

        if cv2.waitKey(0) == 27:
            break
