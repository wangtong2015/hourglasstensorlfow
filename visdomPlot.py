#!/usr/bin/env python
from visdom import Visdom
import numpy as np
import time
viz=Visdom()
assert viz.check_connection(),'visdom server is not connected'
index = 0
class VisdomPlot:
    def __init__(self,xlabel = 'epoch', ylabel = 'loss', title = 'epoch-loss', markersize = 4,legend = '1'):
        global index
        self.name = str(index)
        index+=1
        global viz
        self.viz = viz
        self.data=None
        legend = str(index) if legend == '1' else legend
        self.opts=dict(xlabel=xlabel,ylabel=ylabel,title=title ,markersize = markersize,legend=[legend],width = 800,height = 600)
        self.win = None
    def addData(self,x,y):
        if self.win == None:
            self.data = np.array([[x,y]])
            self.win=self.viz.scatter(self.data,opts=self.opts)
        else:
            self.data = np.vstack((self.data,np.array([[x,y]])))
            self.win=self.viz.scatter(self.data,opts=self.opts,win=self.win)
if __name__=='__main__':
    plot1 = VisdomPlot()
    for i in range(10):
        plot1.addData(i,i+1)
    viz.close()