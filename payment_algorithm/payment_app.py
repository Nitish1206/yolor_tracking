from operator import index
from re import S
from window.payment_window import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QTableWidgetItem,QComboBox,QSizePolicy
import sys
import random
import datetime

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.setupUi(self)
        self.numberofPparking=0
        self.numberofSparking=0
        self.numberofEntries=0
        self.timeEntries=[x for x in range(8*3600,23*3600)]
        self.startendtime=[]
        self.startEnddict={"Start":[],"End":[]}
        self.totalPminute=0
        self.totalSminute=0
        # self.minuteEntries=["%02d" % x for x in range(60)]
        # self.secondsEntries=["%02d" % x for x in range(60)]

        self.dataTabledict={"Row":[],"Start":[],"Finish":[],"Official\nFinish":[],"Ptype":[],
        "Duration":[],"Status":[],"Total P\nHr":[],"Total P\nMin":[],"Total S\nHr":[],"Total S\nMin":[]}
        # self.specificPScalulateButton.clicked.connect(self.prepareData)
        self.addPButton.clicked.connect(self.addPtypeParking)
        self.addSButton.clicked.connect(self.addStypeParking)
        self.clearButton.clicked.connect(self.displayclearEntries)
        self.specificPScalulateButton.clicked.connect(self.displayData)
        self.RandomCalculateButton.clicked.connect(self.createrandompossibilities)
        self.show()
    
    def resetParameters(self):
        self.addPButton.setText("P")
        self.addSButton.setText("S")
        self.numberofPparking=0
        self.numberofSparking=0
        self.numberofEntries=0
        self.totalPminute=0
        self.totalSminute=0
        self.startendtime=[]
        self.startEnddict={"Start":[],"End":[]}
        self.dataTabledict={"Row":[],"Start":[],"Finish":[],"Official\nFinish":[],"Ptype":[],
        "Duration":[],"Status":[],"Total P\nHr":[],"Total P\nMin":[],"Total S\nHr":[],"Total S\nMin":[]}
        # self.displayData()
        self.startEnddict={"Start":[],"End":[],"officialduration":[]}

    def displayclearEntries(self):
        self.resetParameters()
        self.displayData()

    def createrandompossibilities(self):
        numberofEntry=int(self.numberOfParkingLineedit.text())
        self.resetParameters()
        for Entry in range(numberofEntry):
            ptype=random.choice(["P","S"])
            if ptype=="P":
                self.addPtypeParking()
            else:
                self.addStypeParking()

        self.displayData()

    def addPtypeParking(self):
        self.numberofEntries+=1
        self.numberofPparking+=1
        self.addPButton.setText(str(self.numberofPparking)+"P")
        self.dataTabledict["Ptype"].append("P")
        self.dataTabledict["Row"].append(str(self.numberofEntries))
    
    def convert_seconds_to_time(self,user_time_):
        conversion = datetime.timedelta(seconds=user_time_)
        conversion=str(conversion)
        if "," in conversion:
            conversion=conversion.split(",")[-1].strip()
        d = datetime.datetime.strptime(str(conversion), "%H:%M:%S")
        current_frame_time = d.strftime("%H:%M:%S")
        return current_frame_time

    def addStypeParking(self):
        self.numberofEntries+=1
        self.numberofSparking+=1
        self.addSButton.setText(str(self.numberofSparking)+"S")
        self.dataTabledict["Ptype"].append("S")
        self.dataTabledict["Row"].append(str(self.numberofEntries))
    
    def createRandomEntries(self):

        self.dataTabledict["Start"]=[]
        self.dataTabledict["Finish"]=[]
        self.startendtime=[]
        self.startEnddict={"Start":[],"End":[],"officialduration":[]}

        for n in range(2*self.numberofEntries):
            self.startendtime.append(random.choice(self.timeEntries))
            self.startendtime.sort()

        for i,time_ in enumerate(self.startendtime):
            value=self.convert_seconds_to_time(time_)
            if i%2==0 or i==0:
                self.dataTabledict["Start"].append(value)
                self.startEnddict["Start"].append(time_)

            else:
                self.dataTabledict["Finish"].append(value)
                self.startEnddict["End"].append(time_)

    def convertM2HM(self,type,duration,index):
        Hr=duration//60
        M=duration-Hr*60
        if type == "P":
            self.dataTabledict["Total P\nHr"][index]=Hr
            self.dataTabledict["Total P\nMin"][index]=M
            self.dataTabledict["Total S\nHr"][index]=""
            self.dataTabledict["Total S\nMin"][index]=""
        else:
            self.dataTabledict["Total S\nHr"][index]=Hr
            self.dataTabledict["Total S\nMin"][index]=M
            self.dataTabledict["Total P\nHr"][index]=""
            self.dataTabledict["Total P\nMin"][index]=""

    def prepareData(self):
        # self.resetParameters()
        self.createRandomEntries()
        self.dataTabledict["Duration"]=[]
        self.dataTabledict["Status"]=[]
        self.dataTabledict["Total P\nHr"]=[""]*self.numberofEntries
        self.dataTabledict["Total S\nHr"]=[""]*self.numberofEntries
        self.dataTabledict["Total P\nMin"]=[""]*self.numberofEntries
        self.dataTabledict["Total S\nMin"]=[""]*self.numberofEntries
        self.dataTabledict["Official\nFinish"]=[]
        previousofficial=-1
        self.totalPminute=0
        self.totalSminute=0
        officialtime=0
        self.totalPhr=0
        self.totalShr=0
        row=None
        for row in self.dataTabledict["Row"]:
            starttime=self.startEnddict["Start"][int(row)-1]
            endtime=self.startEnddict["End"][int(row)-1]
            ptype=self.dataTabledict["Ptype"][int(row)-1]
            duration= endtime-starttime 
            durationminute=(duration//60) + 1
            self.dataTabledict["Duration"].append(durationminute)
            officailtimefactor=durationminute//60 + 1
            if ptype=="P":
                self.totalPhr+=officailtimefactor*60
                self.startEnddict["officialduration"].append(self.totalPhr)
            else:
                self.totalShr+=officailtimefactor*60
                self.startEnddict["officialduration"].append(self.totalShr)

            officialtimeadder=officailtimefactor*3600
            officialtime=starttime+officialtimeadder
            officialtimevalue=self.convert_seconds_to_time(officialtime)
            self.dataTabledict["Official\nFinish"].append(officialtimevalue)
            
            if previousofficial==-1:
                previousofficial=officialtime
                self.dataTabledict["Status"].append("")
                self.updatetimeHM(ptype,durationminute,int(row)-1)
                
            else:
                if previousofficial > starttime and previousofficial > endtime:
                    self.dataTabledict["Status"].append("within range")
                    previousofficial = officialtime
                    if ptype=="P":
                        self.totalPhr-=officailtimefactor*60
                    else:
                        self.totalShr-=officailtimefactor*60
                    
                    self.startEnddict["officialduration"][-1]-=officailtimefactor*60
                    self.updatetimeHM(ptype,durationminute,int(row)-1)
                
                elif previousofficial > starttime and previousofficial < endtime:
                    self.dataTabledict["Status"].append("start yes end no")
                    previousofficial = officialtime
                    self.updatetimeHM(ptype,durationminute,int(row)-1)
                
                elif previousofficial < starttime:
                    self.dataTabledict["Status"].append("outside range")
                    previousofficial = officialtime
                    if ptype=="P":
                        self.totalPminute = self.startEnddict["officialduration"][int(row)-2]

                    else:
                        self.totalSminute = self.startEnddict["officialduration"][int(row)-2]

                    # self.updatetimeHM(ptype,0,int(row)-2)
                    self.updatetimeHM(ptype,durationminute,int(row)-1)
        if row:
            self.totalPminute=self.startEnddict["officialduration"][int(row)-1]
            
            self.updatetimeHM(ptype,0,int(row)-1)

        print("after update==",self.dataTabledict["Status"])
    def updatetimeHM(self,ptype,durationminute,index):
        if ptype == "P":
            self.totalPminute+=durationminute
            self.convertM2HM("P",self.totalPminute,index)
        else:
            self.totalSminute+=durationminute
            self.convertM2HM("S",self.totalSminute,index)

    def displayData(self):

        self.prepareData()
        row_count=len(self.dataTabledict["Row"])
        column_count=len(list(self.dataTabledict.keys()))
        self.dataTable.setColumnCount(column_count) 
        self.dataTable.setRowCount(row_count)
        self.dataTable.setHorizontalHeaderLabels(list(self.dataTabledict.keys()))

        for row in range(row_count):  # add items from array to QTableWidget
            for col,column in enumerate(self.dataTabledict.keys()):
                try:
                    item = (self.dataTabledict[column][row])
                except:
                    item=""

                item=str(item)
                self.dataTable.setItem(row, col, QTableWidgetItem(item))
        
        self.dataTable.horizontalHeader().setStretchLastSection(True)
        self.dataTable.setWordWrap(True)
        self.dataTable.resizeColumnsToContents()
        # self.dataTable.resizeRowsToContents()

def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()