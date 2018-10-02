import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

import cv2
import pickle
import numpy as np
from keras.models import load_model
from sklearn import preprocessing

from src.preprocess import labelValue, noneBorder
    
class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 300
        self.height = 300
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This is an example button')
        button.move(100,70) 
        button.clicked.connect(self.on_click_bnt)
 
        self.show()
 
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"get Open File Name", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            return fileName
    @pyqtSlot()
    def on_click_bnt (self):
        model = load_model('CNNmodel.h5')
        le = labelValue()
        self.openFileNameDialog()
        # Read the input image
        im = cv2.imread(self.openFileNameDialog())
        im = cv2.resize(im, (1024, 512))

        # pre-process image 
        # blur image to del small noise 
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)

        # Thresh into a binary image 
        # get main line in image 
        thresh = cv2.adaptiveThreshold(
            im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,3)

        # ctrs is Contours 
        # ctrs is an array 2D with all 0 pixel 
        _, ctrs, hier = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #RETR_EXTERNAL

        # boundingRect return a bounding rectangle around the line (contour)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        for rect in rects:
            '''
                a rect have 4 parameter 
                with rect[0], rect[1] is x,y begin of rectangle 
                and rect[2],rect[3] is height and width 
            '''
            if (rect[3] > 80 and rect[2] > 40):
                # we trust wanna get the hand-signal-writing that it must be large 
                # 
                # this code below draw a green rectangle  
                cv2.rectangle(im, (rect[0], rect[1]), (rect[0] +
                                                       rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
                # Make the rectangular region around the sign
                roi = thresh[rect[1]: rect[1]+rect[3], rect[0]: rect[0]+rect[2]]

                if (roi.any()):
                    roi = cv2.resize(roi, (128, 64), interpolation=cv2.INTER_AREA)
                    # roi = cv2.dilate(roi, (3, 3))

                    # process same as the datasets
                    roi = np.array(roi)
                    roi = roi.reshape(128, 64, 1)
                    roi = roi.astype('float32')
                    roi /= 255

                    nbr = model.predict_classes([[roi]])
                    if nbr.size > 0:
                        cv2.putText(im, str(le.inverse_transform(
                            nbr)), (rect[0], rect[1]+50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 254), 2)

        cv2.imwrite('mau_ky.jpg', thresh)
        cv2.imwrite('mau_ky2.jpg', im)
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        k = cv2.waitKey(0) & 0xFF 

        if k == ord('q'):
            cv2.destroyAllWindows()
 
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())