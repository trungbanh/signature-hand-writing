import cv2
import numpy as np

class Imager:
    def __init__(self):
        return None

    def cropImage(self, imP, newPath):
        """
                khung chinh trong khoan 
                h = 147:1750 
                w = 36:1180
                write new image with main frame
        """
        print(imP)
        image = cv2.imread(imP)
        name = imP[imP.find("hinh")+4: imP.find(".png")]
        new = image[147:1750, 36:1180]
        cv2.imwrite(newPath+name+".png", new)

    def frameImage(self, path):
        """
        chia frame cho tung chu ky 
        moi chu ky trong khoan 
        h = 146px
        w = 190px
        return list of frame image 
        """
        x = 0
        y = 0
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 704))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        listFrame = list()
        for w in range(0, 6):
            for h in range(0, 11):
                listFrame.append(img[y:y+64, x:x+128])
                y = h*64
            x = w*128

        return listFrame

    def noneBorder(self, img_path, newPath):
        """
                del all bolder if red = green = blue < 255
        """
        print("read image {0}".format(img_path))
        white = [255, 255, 255]
        image = cv2.imread(img_path)
        name = img_path[img_path.find("hinhCrop")+8: img_path.find(".png")]
        for w in range(len(image[0])):
            for h in range(len(image)):
                r, g, b = image[h, w]
                if r == g and g == b and r < 255:
                    image[h, w] = white
                if (r < 80 and g < 100) or (b < 50 and g < 100):
                    image[h, w] = white
        print("write image {0}".format(newPath+name))
        cv2.imwrite(newPath+name+'.png', image)
