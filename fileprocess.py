import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
# import glob

ch = "wuxiaoyun"
pas = os.listdir(ch)
pa1 = []
for pa in pas:
    if os.path.splitext(pa)[1] == '.mhd':
        pa1.append(pa)


# pa = glob.glob(os.path.join(ch, "*.mhd"))
for pa in pa1:

    # print(pa)
    path = '.' + os.sep + ch + '.' + os.sep + pa
    print(path)
    itkimage = sitk.ReadImage(path) # 读取

    if not os.path.exists('.'+os.sep + 'SaveImagen' +os.sep + ch + '.' + os.sep + pa):
        os.makedirs('.'+os.sep + 'SaveImagen' +os.sep + ch + '.' + os.sep + pa)
    else:
        pass
    image = sitk.GetArrayFromImage(itkimage)

    plt.figure()
    for i in range(len(image)):
        plt.figure(figsize=(10.24, 10.24))
        plt.imshow(image[i,:,:])
        plt.savefig('.'+os.sep + 'SaveImagen' +os.sep+ ch + '.' + os.sep + pa + os.sep + ch + 'image_'+str(i)+'.png')