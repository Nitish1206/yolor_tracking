# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'roi_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1028, 793)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_ = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_.sizePolicy().hasHeightForWidth())
        self.frame_.setSizePolicy(sizePolicy)
        self.frame_.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_.setText("")
        self.frame_.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_.setObjectName("frame_")
        self.verticalLayout_2.addWidget(self.frame_)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.upload_button = QtWidgets.QPushButton(self.widget)
        self.upload_button.setObjectName("upload_button")
        self.horizontalLayout.addWidget(self.upload_button)
        self.clear_button = QtWidgets.QPushButton(self.widget)
        self.clear_button.setObjectName("clear_button")
        self.horizontalLayout.addWidget(self.clear_button)
        self.save_button = QtWidgets.QPushButton(self.widget)
        self.save_button.setObjectName("save_button")
        self.horizontalLayout.addWidget(self.save_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_2.setStretch(1, 1)
        self.horizontalLayout_3.addWidget(self.widget)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.savedtestLabel = QtWidgets.QLabel(self.widget1)
        self.savedtestLabel.setObjectName("savedtestLabel")
        self.horizontalLayout_7.addWidget(self.savedtestLabel)
        self.select_config_combobox = QtWidgets.QComboBox(self.widget1)
        self.select_config_combobox.setObjectName("select_config_combobox")
        self.horizontalLayout_7.addWidget(self.select_config_combobox)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem2)
        self.Parking_type_table_view = QtWidgets.QTableWidget(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Parking_type_table_view.sizePolicy().hasHeightForWidth())
        self.Parking_type_table_view.setSizePolicy(sizePolicy)
        self.Parking_type_table_view.setFrameShape(QtWidgets.QFrame.Box)
        self.Parking_type_table_view.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Parking_type_table_view.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.Parking_type_table_view.setAlternatingRowColors(True)
        self.Parking_type_table_view.setObjectName("Parking_type_table_view")
        self.Parking_type_table_view.setColumnCount(0)
        self.Parking_type_table_view.setRowCount(0)
        self.verticalLayout.addWidget(self.Parking_type_table_view)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem4)
        self.save_parking = QtWidgets.QPushButton(self.widget1)
        self.save_parking.setObjectName("save_parking")
        self.horizontalLayout_4.addWidget(self.save_parking)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.dayselectorLabel = QtWidgets.QLabel(self.widget1)
        self.dayselectorLabel.setObjectName("dayselectorLabel")
        self.horizontalLayout_8.addWidget(self.dayselectorLabel)
        self.daySelectorCombobox = QtWidgets.QComboBox(self.widget1)
        self.daySelectorCombobox.setObjectName("daySelectorCombobox")
        self.horizontalLayout_8.addWidget(self.daySelectorCombobox)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.startTimeLabel = QtWidgets.QLabel(self.widget1)
        self.startTimeLabel.setObjectName("startTimeLabel")
        self.horizontalLayout_2.addWidget(self.startTimeLabel)
        self.startTimeCombobox = QtWidgets.QComboBox(self.widget1)
        self.startTimeCombobox.setFrame(True)
        self.startTimeCombobox.setObjectName("startTimeCombobox")
        self.horizontalLayout_2.addWidget(self.startTimeCombobox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem8)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.hourLabel = QtWidgets.QLabel(self.widget1)
        self.hourLabel.setObjectName("hourLabel")
        self.horizontalLayout_6.addWidget(self.hourLabel)
        self.hourCombobox = QtWidgets.QComboBox(self.widget1)
        self.hourCombobox.setObjectName("hourCombobox")
        self.horizontalLayout_6.addWidget(self.hourCombobox)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem9)
        self.minutesLabel = QtWidgets.QLabel(self.widget1)
        self.minutesLabel.setObjectName("minutesLabel")
        self.horizontalLayout_6.addWidget(self.minutesLabel)
        self.minutesCombobox = QtWidgets.QComboBox(self.widget1)
        self.minutesCombobox.setObjectName("minutesCombobox")
        self.horizontalLayout_6.addWidget(self.minutesCombobox)
        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 2)
        self.horizontalLayout_6.setStretch(2, 1)
        self.horizontalLayout_6.setStretch(3, 2)
        self.horizontalLayout_6.setStretch(4, 3)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        spacerItem10 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem10)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(11, -1, 9, -1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.SaveconfigButton = QtWidgets.QPushButton(self.widget1)
        self.SaveconfigButton.setObjectName("SaveconfigButton")
        self.horizontalLayout_5.addWidget(self.SaveconfigButton)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem11)
        self.saveandstartButton = QtWidgets.QPushButton(self.widget1)
        self.saveandstartButton.setObjectName("saveandstartButton")
        self.horizontalLayout_5.addWidget(self.saveandstartButton)
        self.horizontalLayout_5.setStretch(1, 4)
        self.horizontalLayout_5.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 3)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(7, 1)
        self.verticalLayout.setStretch(9, 1)
        self.verticalLayout.setStretch(11, 1)
        self.verticalLayout.setStretch(13, 1)
        self.horizontalLayout_3.addWidget(self.widget1)
        self.horizontalLayout_3.setStretch(0, 4)
        self.horizontalLayout_3.setStretch(1, 1)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Autonomus Parking Configuration"))
        self.upload_button.setText(_translate("MainWindow", "Upload Image"))
        self.clear_button.setText(_translate("MainWindow", "Clear"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.savedtestLabel.setText(_translate("MainWindow", "Saved_tests"))
        self.save_parking.setText(_translate("MainWindow", "Save"))
        self.dayselectorLabel.setText(_translate("MainWindow", "Select Day"))
        self.startTimeLabel.setText(_translate("MainWindow", "Start Time"))
        self.hourLabel.setText(_translate("MainWindow", "Hour"))
        self.minutesLabel.setText(_translate("MainWindow", "Minutes"))
        self.SaveconfigButton.setText(_translate("MainWindow", "Save"))
        self.saveandstartButton.setText(_translate("MainWindow", "Save and Start"))
