
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:12:16 2019

@author: vince
"""

from pykep import planet, DEG2RAD, epoch, AU, propagate_lagrangian, par2ic, lambert_problem
from math import sqrt, pi, acos, cos, atan,exp
from numpy.linalg import norm
from _Kerbol_System import Moho, Eve, Kerbin, Duna, Jool
from decimal import *
from matplotlib import pyplot as plt
import numpy as np
from scipy import array
from numpy import deg2rad,dot,cross,linalg,rad2deg

KAU = 13599840256 #m


'''
This is a small class for measurement points on the orbit.
Makes things neatier
'''

class OrbitPoint : 
    def __init__(self):
        self.t = 0
        self.r_n = 0
        self.r = [0,0,0]
        self.v_n = 0
        self.v = [0,0,0]        
        self.TAN = 0   
        self.EAN = 0
        self.r_c = 261600000

    def SetTRV(self,t,alt,v_n):
        self.t = self.KTime2KSec(t)
        self.r_n = self.SetRadius(alt)
        self.v_n = v_n

        
    def SetAN(self,SMA,e,sign,radius):
        if radius == "fromVector":
            itRes = 1/e*((SMA/norm(self.r)*(1-e**2))-1)
        if radius == "fromRadius":
            itRes = 1/e*((SMA/self.r_n*(1-e**2))-1)
        #print(itRes)
        if itRes > 1:
            itRes = 1
        TAN = acos(itRes)
        if sign < 0: #sign 
            TAN = -1*TAN
        self.TAN = TAN
        EAN = acos((e+cos(TAN))/(1+e*cos(TAN)))
        if TAN < 0:
            EAN =  -1*EAN
        self.EAN = EAN
    

    def KTime2KSec(self,t):
        year, day, hour, minute, second = t.split(':')
        h2s = 3600
        d2s = 6*h2s
        y2s = 2556.5*h2s
        time = (float(year)-1)*y2s + (float(day)-1)*d2s + float(hour)*h2s + float(minute)*60 + float(second)
        return time
        
    def SetRadius(self,alt):
        r_n = alt + self.r_c
        return r_n
    
    def GetAlt(self):
        alt = self.r_n - self.r_c
        return alt
    
    
class CavemanCraft:
    
    ''' 
        goal
        From caveman data predict r2,v2, at t2 from available data.
        we need to use the launch to get an info on arg Pe
        
        new strategy : calculate TAN at t0, then find a way to backpropagate to tL
        
    '''
    
    def __init__(self):
        # Ugly, but bag it all here
        #self.mu_c = 1.1723328e9*1e9
        self.OrbitPoints = []
        self.mu_c = 1172458888274073600-1.25e5#35*1e12
        self.r_c = 261600000
        self.Pe = 13318146449+self.r_c
        self.Ap = 80146873691+self.r_c
        #self.Pe = 13339605468+self.r_c
        #self.Ap = 21088242027+self.r_c
        self.SMA = (self.Pe+self.Ap)/2
        self.e = 1 - 2*self.Pe/(self.Pe+self.Ap)

        self.N = sqrt(self.mu_c/(self.SMA**3))
     
        self.ArgPe = 0
        self.Mass = 0

        
    def AddOrbitPoint(self, t, r_n, v_n):
        new_OPoint = OrbitPoint()
        self.OrbitPoints.append(new_OPoint)
        self.OrbitPoints[-1].SetTRV(t,r_n,v_n)
        
        
    def KerbinTAN(self,time):
        h2s = 3600
        y2s = 2556.5*h2s
        KTAN = (time/y2s)*2*pi +3.14 -2*pi #will need modulo but hey.
        return KTAN 
    
    def setTime(self, year, day, hour, minute, second):
        h2s = 3600
        d2s = 6*h2s
        y2s = 2556.5*h2s
        time = (year-1)*y2s + (day-1)*d2s + hour*h2s + minute*60 + second 
        #print(time)
        return time
        
    def calcArgPe(self,time):
        #This stuff is likely wrong in some ways
        ArgPe = self.KerbinTAN(time) + (self.OrbitPoints[0].TAN)
        return ArgPe  
   
    def SetAllAN(self,sign):
        for i in range(len(self.OrbitPoints)):
            self.OrbitPoints[i].SetAN(self.SMA,self.e,sign,"fromRadius")

    
    def printVect(self,Vector,Name):
        print(Name,"         "," Norm : ", '{:12}'.format("{0:.0f}".format(norm(Vector))),"[","{0:.0f}".format(Vector[0]),",","{0:.0f}".format(Vector[1]),",","{0:.0f}".format(Vector[2]),"]")

    def deltasAtPoint(self,nR_pred,nV_pred,nR_data,nV_data,Name_r,Name_v):
        delta_nR = np.round((nR_data-nR_pred)/1000,1)
        delta_nV = nV_data-nV_pred
        print("Delta for radius  ",Name_r,"  : ",'{:12}'.format("{0:.0f}".format(delta_nR)),"[km]  ","rel. err. = ","{0:.5f}".format(100*delta_nR/(nR_data/1000)),"%")
        print("Delta for speed   ",Name_v,"  : ",'{:12}'.format("{0:.0f}".format(delta_nV)),"[m/s]","rel. err. = ","{0:.5f}".format(100*delta_nV/nV_data),"%")

    def preciseBurn(self,DV,ISP):
        end_mass = self.Mass/exp(abs(DV)/(9.81*ISP))
        fuel_u = 0.9*(self.Mass - end_mass)/10
        self.Mass = end_mass
        print("Burn for ",DV," [m/s] will require ",np.round(fuel_u,2)," units of fuel")
        
    def calcOrbit(self):
        errEAN_L = 0  *  pi/180
        errArgPe = 0# -3.91  *  pi/180  
        timeErr = 0
        # we try to first convert keplerian elements (assuming ArgPe = 0) to carthesian at t1, then back-propagate to tL to get r, then TAN_L then ArgPe. Hopefully....
        print("------------------------------------------------------")
        #First, fisrt guess at t1 :
        self.OrbitPoints[1].r, self.OrbitPoints[1].v = par2ic([self.SMA,self.e,0,0,0,self.OrbitPoints[1].EAN],self.mu_c)
        #Then back propagate to tL :
        self.OrbitPoints[0].r, self.OrbitPoints[0].v = propagate_lagrangian(self.OrbitPoints[1].r,self.OrbitPoints[1].v,self.OrbitPoints[0].t-self.OrbitPoints[1].t,self.mu_c) #-t0_M should be tL
        self.printVect(self.OrbitPoints[0].r,"rLaunch pred")
        self.printVect(self.OrbitPoints[0].v,"vLaunch pred")        
        print("Launch influence free radius     : ",np.round(self.OrbitPoints[0].r_n/1000,1)," [km]")
        print("Launch radius delta from K orbit : ",np.round(self.OrbitPoints[0].r_n/1000-norm(self.OrbitPoints[0].r)/1000,1)," [km]")
        #we need the sign for the TAN, i.e. are we before or after Pe  Do this with a small delta T
        dt = 1 
        rLdt, vLdt = propagate_lagrangian(self.OrbitPoints[1].r,self.OrbitPoints[1].v,self.OrbitPoints[0].t-self.OrbitPoints[1].t+dt,self.mu_c) #measure dr from dt.
        # if dr is increasing with dt, Pe is past, if decreasing Pe is in front of us
        sign = 1
        if norm(self.OrbitPoints[0].r)-norm(rLdt) < 0 :
            sign = -1
      
        self.OrbitPoints[0].SetAN(self.SMA,self.e,sign,"fromVector")
        self.ArgPe = self.calcArgPe(self.OrbitPoints[0].t)+errArgPe #now that we have a proper radius (virtual radius ignoring kerbin influence at tL, we get a proper trueAnom and can get the ArgPe)
        #self.ArgPe = (17.9)*pi/180
        print("------------------------------------------------------")
        print("TAN at Launch    : ",self.OrbitPoints[0].TAN*180/pi)
        print("EAN at Launch    : ",self.OrbitPoints[0].EAN*180/pi)        
        print("KTAN at Launch   : ",self.KerbinTAN(self.OrbitPoints[0].t)*180/pi)
        print("Arg Pe           : ",self.ArgPe*180/pi)
        print("------------------------------------------------------")
        self.OrbitPoints[2].r, self.OrbitPoints[2].v = propagate_lagrangian(self.OrbitPoints[1].r,self.OrbitPoints[1].v,self.OrbitPoints[2].t-self.OrbitPoints[1].t,self.mu_c) #-t0_M should be tL
        self.printVect(self.OrbitPoints[2].r,"r2 pred from t1")
        self.printVect(self.OrbitPoints[2].v,"v2 pred from t1")   
        self.deltasAtPoint(norm(self.OrbitPoints[2].r),norm(self.OrbitPoints[2].v),self.OrbitPoints[2].r_n,self.OrbitPoints[2].v_n,"r2","v2")
        print("------------------------------------------------------")
        # This all good and well, but the orbit epoch is set to t0, which is unpractical.
        # For lambert solving, it would be much better to have it set to epoch zero i.e. y1d1 00:00:00
        # To do that we need to back propagate to -self.t1, then use r to derive the true anomaly
        self.AddOrbitPoint("0:0:0:0:0",0,0) #Orbit at epoch Zero
        self.OrbitPoints[-1].r, self.OrbitPoints[-1].v = propagate_lagrangian(self.OrbitPoints[1].r,self.OrbitPoints[1].v,-self.OrbitPoints[1].t,self.mu_c) #we now have r,v at epoch Zero
        #print(self.rZero,"    ",self.vZero)

        self.OrbitPoints[-1].SetAN(self.SMA,self.e,-1,"fromVector")        # we are definitly before Pe
        
        
        #self.ArgPe = self.calcArgPe(0)+errArgPe
        print("TAN at Epoch Zero       : ",self.OrbitPoints[-1].TAN*180/pi)
        self.OrbitPoints[-1].r, self.OrbitPoints[-1].v = par2ic([self.SMA,self.e,0,0,self.ArgPe,self.OrbitPoints[-1].EAN],self.mu_c)        
        r0Z, v0Z = propagate_lagrangian(self.OrbitPoints[-1].r,self.OrbitPoints[-1].v,self.OrbitPoints[1].t,self.mu_c)
        self.printVect(r0Z,"r1 pred")
        self.printVect(v0Z,"v1 pred")
        self.deltasAtPoint(norm(r0Z),norm(v0Z),(self.OrbitPoints[1].r_n),(self.OrbitPoints[1].v_n),"r1 fromZero","v1 fromZero")
        r1Z, v1Z = propagate_lagrangian(self.OrbitPoints[-1].r,self.OrbitPoints[-1].v,self.OrbitPoints[2].t,self.mu_c)
        self.printVect(r1Z,"r2 pred")
        self.printVect(v1Z,"v2 pred")          
        self.deltasAtPoint(norm(r1Z),norm(v1Z),(self.OrbitPoints[2].r_n),(self.OrbitPoints[2].v_n),"r2 fromZero","v2 fromZero")
        #Since we have now the Zero vectors, life should be easier for the Lambert part (as the times are consistent now)
        
    def print(self):
        print("------------------------------------------")
        print("Eccentricity     : ",self.e)
        for i in range(len(self.OrbitPoints)):
            print("TAN at t",i,"       : ",self.OrbitPoints[i].TAN*180/pi)
            print("EAN at t",i,"       : ",self.OrbitPoints[i].EAN*180/pi)  



    def correctionBurn(self):
        plt.rcParams['savefig.dpi']=100
        ManT = 76       # manoevre time
        ManT_W = 200     # manoevre window
        Edy2s = 24*3600
        dy2s = 6*3600
        start_epochs = np.arange(ManT,(ManT+ManT_W),0.25)
        ETA = 250
        ETA_W = 250
        duration = np.arange(ETA,(ETA+ETA_W),0.25)


        '''        
        #these are Earth days, to *4 to Kdays (for eph function).

        
        #Sanity checks.
        r2,v2 = Duna.eph(epoch(312.8*0.25)) #check jool position
        print(norm(r2)-self.r_c)
        print(norm(v2))
        
        r1,v1 = propagate_lagrangian(self.OrbitPoints[-1].r,self.OrbitPoints[-1].v,312.8*dy2s,self.mu_c)        
        print(norm(r1)-self.r_c)
        print(norm(v1))       
        '''        
        
        '''
        Solving the lambert problem. the function need times in Edays, so convert later to Kdays.
        Carefull with the Jool ephemeris, since kerbol year starts at y1 d1, substract 1Kday = 0.25Eday
        '''
        
        data = list()
        v_data = list()
        r_data = list()
        v1_data = list()
        for start in start_epochs:
            row = list()
            v_row = list()
            r_row = list()
            v1_row = list()            
            for T in duration:
                #Need to replace the kerbin start point by the ship at time t using
                r1,v1 = propagate_lagrangian(self.OrbitPoints[-1].r,self.OrbitPoints[-1].v,(start)*Edy2s,self.mu_c)
                #r1,v1 = Kerbin.eph(epoch(start))
                r2,v2 = Jool.eph(epoch(start+T))
                l = lambert_problem(r1,r2,T*Edy2s,Kerbin.mu_central_body) #K day = 6h
                DV1 = np.linalg.norm( array(v1)-array(l.get_v1()[0]) )
                v_DV1 = array(v1)-array(l.get_v1()[0]) 

                #DV2 = np.linalg.norm( array(v2)-array(l.get_v2()[0]) )
                #DV1 = max([0,DV1-4000])
                #DV = DV1+DV2
                DV = DV1
                #DV = sqrt(dot(DV1, DV1) + 2 * Kerbin.mu_self / Kerbin.safe_radius) - sqrt(Kerbin.mu_self / Kerbin.safe_radius )
                r_row.append(r1)
                v1_row.append(v1)
                v_row.append(v_DV1)
                row.append(DV)
            data.append(row)
            v_data.append(v_row)
            r_data.append(r_row)
            v1_data.append(v1_row)    

    
        minrows = [min(l) for l in data]
        i_idx = np.argmin(minrows)
        j_idx = np.argmin(data[i_idx])
        best = data[i_idx][j_idx]
        v_best = v_data[i_idx][j_idx]
        r1 = r_data[i_idx][j_idx]
        v1 = v1_data[i_idx][j_idx]
        
        progrd_uv   = -array(v1) / linalg.norm(v1)

        plane_uv   = cross(v1, r1)
        plane_uv   = plane_uv / linalg.norm(plane_uv)
        radial_uv   = cross(plane_uv, progrd_uv)
        EJBK      = sqrt(dot(v_best, v_best) + 2 * Kerbin.mu_central_body / norm(r1)) - sqrt(Kerbin.mu_central_body / norm(r1) )

        progrd_v = dot(progrd_uv, v_best)
        radial_v = dot(radial_uv, v_best)
        
        #print(rad2deg(atan(radial_v/progrd_v)))


        print("TransX escape plan - Kerbin escape")
        print("--------------------------------------")
        print("Best DV: " + str(best))
        print("Best DV heliocentric components:"+str(v_best))
        print("Launch epoch (K-days): " +  str((start_epochs[i_idx])*4    ))
        print("Duration (K-days): " +  str(duration[j_idx]*4))
        print("Prograde:            %10.3f m/s" % np.round(dot(progrd_uv, v_best), 3))
        print("Radial:              %10.3f m/s" % np.round(dot(radial_uv, v_best), 3))
        print("Planar:              %10.3f m/s" % np.round(dot(plane_uv, v_best), 3))
        print("Hyp. excess velocity:%10.3f m/s" % np.round(sqrt(dot(v_best, v_best)), 3))
        #print("Earth escape burn:   %10.3f m/s" % np.round(EJBK, 3))


        duration_pl, start_epochs_pl = np.meshgrid(duration, start_epochs)
        plt.contour(start_epochs_pl*4,duration_pl*4,array(data),levels = list(np.linspace(best,3000,12)))
        #plt.imshow(array(data).T, cmap=plt.cm.rainbow, origin = "lower", vmin = best, vmax = 5000, extent=[0.0, 850, 10, 470.0], interpolation='bilinear')

        #plt.colorbar(im);
        plt.colorbar()
        plt.show()
        
        
    def fineCorrectionBurn(self):
        #Propagate orbit to correction burn 1, then apply it to orbit, then propagage further and check.
        #If okay, then add in the manual inclination correction, and then do a final Lambert.
        self.AddOrbitPoint("1:319:0:0:0",14646793325,9353.0)            
        rMan1, vMan1 = propagate_lagrangian(self.OrbitPoints[-2].r,self.OrbitPoints[-2].v,self.OrbitPoints[-1].t,self.mu_c)
        print(str(vMan1))
        vMan1 = np.asarray(vMan1)
        vMan1 += [-34.8,35.9,0] #?????? WTF norm ok but direction is wrong (apprently)
        #vMan1 += [-25.87664536,+42.44449723,0]
        print(str(vMan1))
        self.OrbitPoints[-1].r = rMan1
        self.OrbitPoints[-1].r_n = norm(rMan1)
        self.OrbitPoints[-1].v = vMan1
        self.OrbitPoints[-1].v_n = norm(vMan1)       
        
        #Propagate to a new point         
        self.AddOrbitPoint("1:399:3:33:30",17739170771,7823.5)
        rCheck, vCheck = propagate_lagrangian(self.OrbitPoints[-2].r,self.OrbitPoints[-2].v,self.OrbitPoints[-1].t-self.OrbitPoints[-2].t,self.mu_c)
        self.printVect(rCheck,"r2 pred")
        self.printVect(vCheck,"v2 pred")          
        self.deltasAtPoint(norm(rCheck),norm(vCheck),(self.OrbitPoints[-1].r_n),(self.OrbitPoints[-1].v_n),"r check fromZero","v check fromZero")
        self.AddOrbitPoint("1:417:2:58:04",18284938767,7574.6)
        rCheck, vCheck = propagate_lagrangian(self.OrbitPoints[-3].r,self.OrbitPoints[-3].v,self.OrbitPoints[-1].t-self.OrbitPoints[-3].t,self.mu_c)
        self.printVect(rCheck,"r3 pred")
        self.printVect(vCheck,"v3 pred")          
        self.deltasAtPoint(norm(rCheck),norm(vCheck),(self.OrbitPoints[-1].r_n),(self.OrbitPoints[-1].v_n),"r check2 fromZero","v check2 fromZero")
        
        #self.OrbitPoints[-3].r, self.OrbitPoints[-3].v = propagate_lagrangian(self.OrbitPoints[-3].r,self.OrbitPoints[-3].v,0,self.mu_c)
        
        plt.rcParams['savefig.dpi']=100
        ManT = 76       # manoevre time
        ManT_W = 5     # manoevre window
        Edy2s = 24*3600
        dy2s = 6*3600
        start_epochs = np.arange(ManT,(ManT+ManT_W),0.25)
        ETA = 20
        ETA_W = 200  
        duration = np.arange(ETA,(ETA+ETA_W),0.25)


      
        data = list()
        v_data = list()
        r_data = list()
        v1_data = list()
        for start in start_epochs:
            row = list()
            v_row = list()
            r_row = list()
            v1_row = list()            
            for T in duration:
                #Need to replace the kerbin start point by the ship at time t using
                r1,v1 = propagate_lagrangian(self.OrbitPoints[-3].r,self.OrbitPoints[-3].v,(start)*Edy2s-self.OrbitPoints[-3].t,self.mu_c)
                #r1,v1 = Kerbin.eph(epoch(start))
                r2,v2 = Duna.eph(epoch(start+T))
                l = lambert_problem(r1,r2,T*Edy2s,Kerbin.mu_central_body) #K day = 6h
                DV1 = np.linalg.norm( array(v1)-array(l.get_v1()[0]) )
                v_DV1 = array(v1)-array(l.get_v1()[0]) 

                #DV2 = np.linalg.norm( array(v2)-array(l.get_v2()[0]) )
                #DV1 = max([0,DV1-4000])
                #DV = DV1+DV2
                DV = DV1
                #DV = sqrt(dot(DV1, DV1) + 2 * Kerbin.mu_self / Kerbin.safe_radius) - sqrt(Kerbin.mu_self / Kerbin.safe_radius )
                r_row.append(r1)
                v1_row.append(v1)
                v_row.append(v_DV1)
                row.append(DV)
            data.append(row)
            v_data.append(v_row)
            r_data.append(r_row)
            v1_data.append(v1_row)    

    
        minrows = [min(l) for l in data]
        i_idx = np.argmin(minrows)
        j_idx = np.argmin(data[i_idx])
        best = data[i_idx][j_idx]
        v_best = v_data[i_idx][j_idx]
        r1 = r_data[i_idx][j_idx]
        v1 = v1_data[i_idx][j_idx]
        
        progrd_uv   = -array(v1) / linalg.norm(v1)

        plane_uv   = cross(v1, r1)
        plane_uv   = plane_uv / linalg.norm(plane_uv)
        radial_uv   = cross(plane_uv, progrd_uv)
        EJBK      = sqrt(dot(v_best, v_best) + 2 * Kerbin.mu_central_body / norm(r1)) - sqrt(Kerbin.mu_central_body / norm(r1) )

        progrd_v = dot(progrd_uv, v_best)
        radial_v = dot(radial_uv, v_best)
        
        #print(rad2deg(atan(radial_v/progrd_v)))


        print("TransX escape plan - Kerbin escape")
        print("--------------------------------------")
        print("Best DV: " + str(best))
        print("Best DV heliocentric components:"+str(v_best))
        print("Launch epoch (K-days): " +  str((start_epochs[i_idx])*4    ))
        print("Duration (K-days): " +  str(duration[j_idx]*4))
        print("Prograde:            %10.3f m/s" % np.round(dot(progrd_uv, v_best), 3))
        print("Radial:              %10.3f m/s" % np.round(dot(radial_uv, v_best), 3))
        print("Planar:              %10.3f m/s" % np.round(dot(plane_uv, v_best), 3))
        print("Hyp. excess velocity:%10.3f m/s" % np.round(sqrt(dot(v_best, v_best)), 3))
        #print("Earth escape burn:   %10.3f m/s" % np.round(EJBK, 3))


        duration_pl, start_epochs_pl = np.meshgrid(duration, start_epochs)
        plt.contour(start_epochs_pl*4,duration_pl*4,array(data),levels = list(np.linspace(best,3000,12)))
        #plt.imshow(array(data).T, cmap=plt.cm.rainbow, origin = "lower", vmin = best, vmax = 5000, extent=[0.0, 850, 10, 470.0], interpolation='bilinear')

        #plt.colorbar(im);
        plt.colorbar()
        plt.show()        
        
        
        #hmmm, lets rescale the pred vector to the norm of observation :
        

Craft = CavemanCraft()

#Data for test1
#self.Pe = 13339605468+self.r_c
#self.Ap = 21088242027+self.r_c
#TestNav.AddOrbitPoint("1:230:0:26:44",13338420256,0)
#TestNav.AddOrbitPoint("1:255:4:48:3",13588012231,10109.9)
#TestNav.AddOrbitPoint("1:313:2:31:17",15314749933,9134.7)


Craft.AddOrbitPoint("1:229:02:39:34",13338240256,0)
Craft.AddOrbitPoint("1:265:04:56:23",14335339555,11648.2)
Craft.AddOrbitPoint("1:304:00:03:58",17486647053,10351.8)

Craft.SetAllAN(1)
#TestNav.print()       
Craft.calcOrbit()
Craft.correctionBurn()
#Craft.fineCorrectionBurn()
Craft.Mass = 25980
Craft.preciseBurn(-14.737,350)
Craft.preciseBurn(-6.901,350)
Craft.preciseBurn(27.121,350)
#plot_innerKerbol(epoch(100))