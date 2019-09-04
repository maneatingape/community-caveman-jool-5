# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:12:16 2019

@author: vince
"""

from pykep import planet, DEG2RAD, epoch, AU, propagate_lagrangian, par2ic, lambert_problem
from math import sqrt, pi, acos, cos, atan
from numpy.linalg import norm
from _Kerbol_System import Moho, Eve, Kerbin, Duna, Jool
from decimal import *
from matplotlib import pyplot as plt
import numpy as np
from scipy import array
from numpy import deg2rad,dot,cross,linalg,rad2deg

KAU = 13599840256 #m


    
class CavemanNavigator:
    
    ''' 
        goal
        From caveman data predict r2,v2, at t2 from available data.
        we need to use the launch to get an info on arg Pe
        
        HOW IT WORKS : 
            
            1) note time of Launch tL
            2) at two further times (t0,t1) out of kerbin SOI, take altitude, time and speed
            3) Create a dummy orbit from t0 data with ArgPe set to zero
            4) Backpropagate the dummy orbit from t0 to tL. Calculate the virtual radius (altitude without Kerbin influence)
            5) From the time tL calculate the Kerbin True Anomaly and the ship True Anomaly. From those, compute the ArgPe
            6) With the complete orbit info, do a check with the t1 point
            7) Back propagate the orbit to Epoch Zero (y1d1)
            8) Solve the Lambert problem for a set of ship positions and Jool ETAs.
            9) Select best correction burn
            10) Profit
    
        TODO : create a class for the data points
        
    '''
    
    def __init__(self):
        # Ugly, but bag it all here
        #self.mu_c = 1.1723328e9*1e9
        self.mu_c = 1172458888274073600#35*1e12
        self.r_c = 261600000
        self.Pe = 13339382571+self.r_c
        self.Ap = 71600522497+self.r_c
        self.SMA = (self.Pe+self.Ap)/2
        self.e = 1 - 2*self.Pe/(self.Pe+self.Ap)
        self.tL = self.setTime(1,229,4,12,23)
        self.t0 = self.setTime(1,231,0,34,15)
        self.t1 = self.setTime(1,323,3,27,15)
        self.t0_M = self.t0-self.tL
        self.t1_M = self.t1-self.tL
        self.rnL = 13338420256+self.r_c
        self.rn0 = 13341111200+self.r_c
        self.rn1 = 19701742109+self.r_c
        self.rL = [0,0,0]
        self.r0 = [0,0,0]
        self.r1 = [0,0,0]
        self.vn0 = 12044.7
        self.vn1 = 9487.6
        self.vL = [0,0,0]
        self.v0 = [0,0,0]
        self.v1 = [0,0,0]
        self.vZero = [0,0,0]
        self.rZero = [0,0,0]        
        self.TAN_Zero = 0
        self.EAN_Zero = 0
        self.N = sqrt(self.mu_c/(self.SMA**3))
        self.TAN_L = self.calcTAN(self.rnL,1)
        self.TAN_t0 = self.calcTAN(self.rn0,1)
        self.TAN_t1 = self.calcTAN(self.rn1,1)
        self.EAN_L = self.calcEAN(self.TAN_L)
        self.EAN_t0 = self.calcEAN(self.TAN_t0)
        self.EAN_t1 = self.calcEAN(self.TAN_t1)        
        self.ArgPe = 0

        
        
        
    def KerbinTAN(self,time):
        h2s = 3600
        y2s = 2556.5*h2s
        KTAN = (time/y2s)*2*pi +3.14 #will need modulo but hey.
        return KTAN
     

    
    def setTime(self, year, day, hour, minute, second):
        h2s = 3600
        d2s = 6*h2s
        y2s = 2556.5*h2s
        time = (year-1)*y2s + (day-1)*d2s + hour*h2s + minute*60 + second 
        return time
        
    def calcTAN(self,radius,sign):
        itRes = 1/self.e*((self.SMA/radius*(1-self.e**2))-1)
        #print(itRes)
        if itRes > 1:
            itRes = 1
        TrueAnomaly = acos(itRes)
        if sign < 0: #sign 
            TrueAnomaly = -1*TrueAnomaly
        return TrueAnomaly

    def calcArgPe(self,time):
        #This stuff is likely wrong in some ways
        ArgPe = self.KerbinTAN(time) - (self.TAN_L)
        return ArgPe
    
    def calcEAN(self,TAN):
        EAN = acos((self.e+cos(TAN))/(1+self.e*cos(TAN)))
        if TAN < 0:
            EAN =  -1*EAN
        return EAN

    
    def printVect(self,Vector,Name):
        print(Name,"         "," Norm : ", '{:12}'.format("{0:.0f}".format(norm(Vector))),"[","{0:.0f}".format(Vector[0]),",","{0:.0f}".format(Vector[1]),",","{0:.0f}".format(Vector[2]),"]")

    def deltasAtPoint(self,nR_pred,nV_pred,nR_data,nV_data,Name_r,Name_v):
        delta_nR = nR_data-nR_pred
        delta_nV = nV_data-nV_pred
        print("Delta for radius  ",Name_r,"  : ",'{:12}'.format("{0:.0f}".format(delta_nR)),"[m]  ","rel. err. = ","{0:.5f}".format(100*delta_nR/nR_data),"%")
        print("Delta for speed   ",Name_v,"  : ",'{:12}'.format("{0:.0f}".format(delta_nV)),"[m/s]","rel. err. = ","{0:.5f}".format(100*delta_nV/nV_data),"%")

        
    def calcOrbit(self):
        errEAN_L = 0  *  pi/180
        errArgPe = 0  *  pi/180  
        timeErr = 0
        # we try to first convert keplerian elements (assuming ArgPe = 0) to carthesian at t1, then back-propagate to tL to get r, then TAN_L then ArgPe. Hopefully....
        print("------------------------------------------------------")
        #First, fisrt guess at t1 :
        self.r0, self.v0 = par2ic([self.SMA,self.e,0,0,0,self.EAN_t0],self.mu_c)
        #Then back propagate to tL :
        self.rL, self.vL = propagate_lagrangian(self.r0,self.v0,-self.t0_M,self.mu_c) #-t0_M should be tL
        self.printVect(self.rL,"rL pred")
        self.printVect(self.vL,"vL pred")        
        print("rL influence free radius     : ",self.rnL)
        print("rL radius delta from K orbit : ",self.rnL-norm(self.rL))
        #We need the sign for the TAN, i.e. are we before or after Pe  Do this with a small delta T
        dt = 1 
        rLdt, vLdt = propagate_lagrangian(self.r0,self.v0,-self.t0_M+dt,self.mu_c) #measure dr from dt.
        # if dr is increasing with dt, Pe is past, if decreasing Pe is in front of us
        sign = 1
        if norm(self.rL)-norm(rLdt) < 0 :
            sign = -1
      
        self.TAN_L = self.calcTAN(norm(self.rL),sign)
        self.EAN_L = self.calcEAN(self.TAN_L)
        self.ArgPe = self.calcArgPe(self.tL)+errArgPe #now that we have a proper radius (virtual radius ignoring kerbin influence at tL, we get a proper trueAnom and can get the ArgPe)
        print("------------------------------------------------------")
        print("TAN at Launch    : ",self.TAN_L*180/pi)
        print("EAN at Launch    : ",self.EAN_L*180/pi)        
        print("KTAN at Launch   : ",self.KerbinTAN(self.tL)*180/pi)
        print("Arg Pe           : ",self.ArgPe*180/pi)
        print("------------------------------------------------------")
        self.r1, self.v1 = propagate_lagrangian(self.r0,self.v0,self.t1-self.t0,self.mu_c) #-t0_M should be tL
        self.printVect(self.r1,"r1 pred")
        self.printVect(self.v1,"v1 pred")   
        self.deltasAtPoint(norm(self.r1),norm(self.v1),self.rn1,self.vn1,"r1","v1")
        # This all good and well, but the orbit epoch is set to t0, which is unpractical.
        # For lambert solving, it would be much better to have it set to epoch zero i.e. y1d1 00:00:00
        # To do that we need to back propagate to -self.t1, then use r to derive the true anomaly
        self.rZero, self.vZero = propagate_lagrangian(self.r0,self.v0,-self.t0,self.mu_c) #we now have r,v at epoch Zero
        print(self.rZero,"    ",self.vZero)
        self.TAN_Zero = self.calcTAN(norm(self.rZero),-1)   # we are definitly before Pe
        EAN_ZeroErr = 0*pi/180 ### There is a problem with E at epoch
        self.EAN_Zero = self.calcEAN(self.TAN_Zero)+EAN_ZeroErr
        #self.ArgPe = self.calcArgPe(0)+errArgPe
        print("TAN at Epoch Zero       : ",self.TAN_Zero*180/pi)
        self.rZero, self.vZero = par2ic([self.SMA,self.e,0,0,self.ArgPe,self.EAN_Zero],self.mu_c)        
        r0Z, v0Z = propagate_lagrangian(self.rZero,self.vZero,self.t0,self.mu_c)
        self.printVect(r0Z,"r0 pred")
        self.printVect(v0Z,"v0 pred")
        self.deltasAtPoint(norm(r0Z),norm(v0Z),(self.rn0),(self.vn0),"r0fromZero","v0fromZero")
        r1Z, v1Z = propagate_lagrangian(self.rZero,self.vZero,self.t1,self.mu_c)
        self.printVect(r1Z,"r1 pred")
        self.printVect(v1Z,"v1 pred")          
        self.deltasAtPoint(norm(r1Z),norm(v1Z),(self.rn1),(self.vn1),"r1fromZero","v1fromZero")
        #Since we have now the Zero vectors, life should be easier for the Lambert part (as the times are consistent now)
        
    def print(self):
        print("------------------------------------------")
        print("Eccentricity     : ",self.e)
        print("TAN at t0        : ",self.TAN_t0*180/pi)
        print("EAN at t0        : ",self.EAN_t0*180/pi)  
        print("TAN at t1        : ",self.TAN_t1*180/pi)
        print("EAN at t1        : ",self.EAN_t1*180/pi)  



    def correctionBurn(self):
        plt.rcParams['savefig.dpi']=100
        ManT = 80       # manoevre time
        ManT_W = 5     # manoevre window
        Edy2s = 24*3600
        start_epochs = np.arange(ManT,(ManT+ManT_W),0.25)
        ETA = 125
        ETA_W = 500
        duration = np.arange(ETA,(ETA+ETA_W),0.25)


        
        #these are Earth days, to *4 to Kdays (for eph function).

        '''
        #Sanity checks.
        r2,v2 = Jool.eph(epoch(322.5*0.25)) #check jool position
        print(norm(r2))
        print(norm(v2))
        r1,v1 = propagate_lagrangian(self.rZero,self.vZero,323.125*dy2s,self.mu_c)        
        print(norm(r1))
        print(norm(v1))       
        '''
        
        '''
        Solving the lambert problem. the function need times in Edays, so convert later to Kdays.
        Carefull with the Jool ephemeris, since kerbol year starts at y1 d1, substract 1Kday = 0.25Eday
        '''
        
        data = list()
        v_data = list()
        for start in start_epochs:
            row = list()
            v_row = list()
            for T in duration:
                #Need to replace the kerbin start point by the ship at time t using
                r1,v1 = propagate_lagrangian(self.rZero,self.vZero,(start)*Edy2s,self.mu_c)
                #r1,v1 = Kerbin.eph(epoch(start))
                r2,v2 = Jool.eph(epoch(start+T-0.25))
                l = lambert_problem(r1,r2,T*Edy2s,Kerbin.mu_central_body) #K day = 6h
                DV1 = np.linalg.norm( array(v1)-array(l.get_v1()[0]) )
                v_DV1 = array(v1)-array(l.get_v1()[0]) 
                #DV2 = np.linalg.norm( array(v2)-array(l.get_v2()[0]) )
                #DV1 = max([0,DV1-4000])
                #DV = DV1+DV2
                DV = DV1
                #DV = sqrt(dot(DV1, DV1) + 2 * Kerbin.mu_self / Kerbin.safe_radius) - sqrt(Kerbin.mu_self / Kerbin.safe_radius )
                v_row.append(v_DV1)
                row.append(DV)
            data.append(row)
            v_data.append(v_row)
    

    
        minrows = [min(l) for l in data]
        i_idx = np.argmin(minrows)
        j_idx = np.argmin(data[i_idx])
        best = data[i_idx][j_idx]
        v_best = v_data[i_idx][j_idx]
        
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
        print("Launch epoch (K-days): " +  str((start_epochs[i_idx])*4-1    ))
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

TestNav = CavemanNavigator()
TestNav.print()       
TestNav.calcOrbit()
TestNav.correctionBurn()

#plot_innerKerbol(epoch(100))