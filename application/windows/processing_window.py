# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'processing_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1332, 874)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.video_processor = QtWidgets.QLabel(Form)
        self.video_processor.setMinimumSize(QtCore.QSize(1080, 720))
        self.video_processor.setFrameShape(QtWidgets.QFrame.Box)
        self.video_processor.setText("")
        self.video_processor.setObjectName("video_processor")
        self.verticalLayout.addWidget(self.video_processor)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.progressBar = QtWidgets.QProgressBar(Form)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout.addWidget(self.progressBar)
        self.uploadvideo = QtWidgets.QPushButton(Form)
        self.uploadvideo.setObjectName("uploadvideo")
        self.horizontalLayout.addWidget(self.uploadvideo)
        self.start_processing = QtWidgets.QPushButton(Form)
        self.start_processing.setObjectName("start_processing")
        self.horizontalLayout.addWidget(self.start_processing)
        self.stop_processing = QtWidgets.QPushButton(Form)
        self.stop_processing.setObjectName("stop_processing")
        self.horizontalLayout.addWidget(self.stop_processing)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout.setStretch(0, 4)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.detail_table = QtWidgets.QTableWidget(Form)
        self.detail_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.detail_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.detail_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.detail_table.setAutoScroll(False)
        self.detail_table.setWordWrap(False)
        self.detail_table.setObjectName("detail_table")
        self.detail_table.setColumnCount(0)
        self.detail_table.setRowCount(0)
        self.horizontalLayout_2.addWidget(self.detail_table)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.uploadvideo.setText(_translate("Form", "Uploadvideo"))
        self.start_processing.setText(_translate("Form", "Start"))
        self.stop_processing.setText(_translate("Form", "Stop"))
