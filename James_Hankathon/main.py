import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from NasaAPICrawler import searchText
from StylizeImg import stylizeImg

assets_path = 'assets/'


class JTHankSolution(QMainWindow):
    def __init__(self):
        super().__init__()
        self.text = ""
        self.style_num = 0
        self.styles = [assets_path + self.text + ".jpg", assets_path + "style1_pic.jpg", 
                  assets_path + "style2_pic.jpg", assets_path + "style3_pic.jpg", 
                  assets_path + "style4_pic.jpg", assets_path + "style5_pic.jpg", 
                  assets_path + "texture1_pic.jpg", assets_path + "texture2_pic.jpg", 
                  assets_path + "texture3_pic.jpg", assets_path + "texture4_pic.jpg"]
        self.window = QWidget()
        self.window.setWindowTitle("James Hankathon Solution to Nasa Space Apps Challenge 2022")
        self.layout = QVBoxLayout()
        self.horizontal = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.input_mes = QLabel("Please input text you wish to search~")
        self.search_bar.setMaximumHeight(30)
        self.go_btn = QPushButton("Go")
        self.go_btn.setMinimumHeight(30)
        self.go_btn.clicked.connect(self.search)
        self.style_btn = QPushButton("Change Style")
        self.style_btn.setMinimumHeight(30)
        self.style_btn.clicked.connect(self.changeStyle)
        #self.forward_btn = QPushButton(">")
        #self.forward_btn.setMinimumHeight(30)
        self.pic_label = QLabel()
        self.horizontal.addWidget(self.input_mes)
        self.horizontal.addWidget(self.search_bar)
        self.horizontal.addWidget(self.go_btn)
        self.horizontal.addWidget(self.style_btn)
        #self.horizontal.addWidget(self.forward_btn)
        self.layout.addLayout(self.horizontal)
        self.layout.addWidget(self.pic_label)
        self.window.setLayout(self.layout)
        self.window.show()
        
    def search(self):
        self.text = self.search_bar.text()
        if self.text == "":
            return
        searchText(self.text)
        spaceImg = "assets/" + self.text + ".jpg"
        pixmap = QPixmap(spaceImg)
        self.pic_label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())
        self.style_num = 0
    
    def changeStyle(self):
        if self.text == "":
            return
        else:
            if self.style_num == 9:
                self.style_num == 0
            else:
                self.style_num += 1;
        #spaceImg = "assets/" + self.text + ".jpg"
        #spaceImg = stylizeImg(spaceImg, self.styles[self.style_num])
        spaceImg = self.styles[self.style_num]
        pixmap = QPixmap(spaceImg)
        self.pic_label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())
    
        
def launch_demo():
    app = QApplication([])
    window = JTHankSolution()
    app.exec()


if __name__ == '__main__':
    #demo()
    launch_demo()
    
    