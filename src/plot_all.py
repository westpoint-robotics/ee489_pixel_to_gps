#!/usr/bin/python

import matplotlib.pyplot as pp
import os


raw = []
for filename in os.listdir("/home/wborn/course"):
    fp = open("/home/wborn/course/"+filename,'r')
    raw.append(fp.read().split(';'))
    fp.close()


points = []
for i in raw:
    temp=[]
    for j in i:
        if j.split(',') != ['']:
            temp.append(j.split(','))
    points.append(temp)
points=[x for x in points if x != ['']]

for j in range(len(points)):
    for i in range(len(points[j])-1):
        pp.plot([float(points[j][i][0]),float(points[j][i+1][0])],[float(points[j][i][1]),float(points[j][i+1][1])], '-',color='black',linewidth=7.0)
    #pp.draw()
    #pp.pause(0.000001)

raw=[]

for filename in os.listdir("/home/wborn/trials"):
    fp = open("/home/wborn/trials/"+filename,'r')
    raw.append(fp.read().split(';'))
    fp.close()


points = []
for i in raw:
    temp=[]
    for j in i:
        if j.split(',') != ['']:
            temp.append(j.split(','))
    points.append(temp)
points=[x for x in points if x != ['']]

for j in range(len(points)):
    for i in range(len(points[j])-1):
        pp.plot([float(points[j][i][0]),float(points[j][i+1][0])],[float(points[j][i][1]),float(points[j][i+1][1])], 'r-')
    #pp.draw()
    #pp.pause(0.000001)

raw = []
for filename in os.listdir("/home/wborn/auto_trials"):
    fp = open("/home/wborn/auto_trials/"+filename,'r')
    raw.append(fp.read().split(';'))
    fp.close()


points = []
for i in raw:
    temp=[]
    for j in i:
        if j.split(',') != ['']:
            temp.append(j.split(','))
    points.append(temp)
points=[x for x in points if x != ['']]

for j in range(len(points)):
    for i in range(len(points[j])-1):
        pp.plot([float(points[j][i][0]),float(points[j][i+1][0])],[float(points[j][i][1]),float(points[j][i+1][1])], 'g-')
    #pp.draw()
    #pp.pause(0.000001)
pp.show()
