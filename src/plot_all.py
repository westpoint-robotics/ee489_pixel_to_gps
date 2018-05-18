#!/usr/bin/python

import matplotlib.pyplot as pp
import os

raw=''

for filename in os.listdir("/home/wborn/trials"):
    fp = open("/home/wborn/trials/"+filename,'r')
    raw+=fp.read()
    fp.close()

pairs=raw.split(';')
points = []
for i in pairs:
    points.append(i.split(','))
points=[x for x in points if x != ['']]

for p in points:
    pp.plot(float(p[0]),float(p[1]), 'bo')
    #pp.draw()
    #pp.pause(0.000001)

pp.show()

print points
