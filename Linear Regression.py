#!/usr/bin/env python
# coding: utf-8

# In[4]:


from matplotlib import pyplot as plt
from matplotlib import dates as md
import numpy as np
from numpy import polyfit
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

#import data to dataframe
data = pd.read_excel(r'C:\Users\Maria Munir Stokes\Desktop\Smart Meter Data\KMeansData.xlsx',13)
#Plot
#plot customisation
#plt.figure(figsize = (25,10))
#plt.scatter(data['Number of Residents'],data['Mean Daily Usage(year)'])
#plt.xticks(fontweight = 'bold')
#plt.title("Number of Residents Vs. Average Yearly Energy Consumption(kWh) ",fontweight =  'bold',fontsize = 20)
#plt.xlabel(" Number of Residents",fontweight =  'bold',fontsize = 18)
#plt.ylabel("Mean Yearly Energy Consumption (kWh)",fontweight =  'bold',fontsize = 18)


resident_df =[1,2,3,4,5,6,7,8,9]
no_outliarsresident_df=[1,2,3,4,5,6,7]
#array to count number of samples in each category
samplenumber_df = [data[data['Number of Residents']==1].count(),data[data['Number of Residents']==2].count(),data[data['Number of Residents']==3].count(),data[data['Number of Residents']==4].count(),data[data['Number of Residents']==5].count(),data[data['Number of Residents']==6].count(),data[data['Number of Residents']==7].count(),data[data['Number of Residents']==8].count(),data[data['Number of Residents']==9].count()]



#creating separate dataframes to work out
resident1 = data.loc[data['Number of Residents']== 1]
resident2 = data.loc[data['Number of Residents']== 2]
resident3 = data.loc[data['Number of Residents']== 3]
resident4 = data.loc[data['Number of Residents']== 4]
resident5 = data.loc[data['Number of Residents']== 5]
resident6 = data.loc[data['Number of Residents']== 6]
resident7 = data.loc[data['Number of Residents']== 7]
resident8 = data.loc[data['Number of Residents']== 8]
resident9 = data.loc[data['Number of Residents']== 9]

#working out the mean of each of the different values of residents
mean1 = resident1['Mean Daily Usage(year)'].mean()
mean2 = resident2['Mean Daily Usage(year)'].mean()
mean3 = resident3['Mean Daily Usage(year)'].mean()
mean4 = resident4['Mean Daily Usage(year)'].mean()
mean5 = resident5['Mean Daily Usage(year)'].mean()
mean6 = resident6['Mean Daily Usage(year)'].mean()
mean7 = resident7['Mean Daily Usage(year)'].mean()
mean8 = resident8['Mean Daily Usage(year)'].mean()
mean9 = resident9['Mean Daily Usage(year)'].mean()


mean_df = [mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8,mean9]
no_outliars_mean =[mean1,mean2,mean3,mean4,mean5,mean6,mean7]
print('mean:',no_outliars_mean)


#with outliars
plt.figure(figsize = (4,3))
plota = plt.plot(resident_df,mean_df,'o')
plt.xticks(resident_df,fontweight = 'bold')
plt.title("Number of Residents Vs. Average Yearly Energy Consumption (kWh) ",fontweight =  'bold',fontsize = 8)
plt.xlabel(" Number of Residents",fontweight =  'bold',fontsize = 7)
plt.ylabel("Mean Yearly Energy Consumption (kWh)",fontweight =  'bold',fontsize = 7)

m1,c1 =np.polyfit(resident_df,mean_df,1)#working out the parameters of the best fit line
#print('m1',m1) #gradient
#print('c1',c1) #intercept
#creating dataframe with best fit values as i cannot multiply the current data frame by it
bfresident_df =[1,2,3,4,5,6,7,8,9]
bestfit_df =[(1*m1+c1),(2*m1+c1),(3*m1+c1),(4*m1+c1),(5*m1+c1),(6*m1+c1),(7*m1+c1),(8*m1+c1),(9*m1+c1)]

plt.plot(bfresident_df,bestfit_df)


#without outliars
plt.figure(figsize = (4,3))
plotb = plt.plot(no_outliarsresident_df,no_outliars_mean,'o')
plt.xticks(no_outliarsresident_df,fontweight = 'bold')
plt.title("Number of Residents Vs. Average Yearly Energy Consumption (kWh) ",fontweight =  'bold',fontsize = 8)
plt.xlabel("Number of Residents",fontweight =  'bold',fontsize = 7)
plt.ylabel("Mean Yearly Energy Consumption (kWh)",fontweight =  'bold',fontsize = 7)
m2,c2 =np.polyfit(no_outliarsresident_df,no_outliars_mean,1) #working out the parameters of the best fit line
#print('m2',m2) #gradient
#print("c2",c2) #intercept
no_outliar_bestfit_df =[(1*m2+c2),(2*m2+c2),(3*m2+c2),(4*m2+c2),(5*m2+c2),(6*m2+c2),(7*m2+c2)]
plt.plot(no_outliarsresident_df,no_outliar_bestfit_df)

#creating the log plot
def logcurvefunction(no_outliarsresident_df,a,b,c):
    return(a*np.ln(b+no_outliarsresident_df)+c)

popt,pcov = curve_fit(logcurvefunction,no_outliarsresident_df,no_outliars_mean)
print("popt",popt)#optimum values for abc
print("pcov",pcov)
plt.figure(figsize = (4,3))
plotb = plt.plot(no_outliarsresident_df,no_outliars_mean,'o')
plt.xticks(no_outliarsresident_df,fontweight = 'bold')
plt.title("Number of Residents Vs. Average Yearly Energy Consumption (kWh) ",fontweight =  'bold',fontsize = 8)
plt.xlabel("Number of Residents",fontweight =  'bold',fontsize = 7)
plt.ylabel("Mean Yearly Energy Consumption (kWh)",fontweight =  'bold',fontsize = 7)
plt.plot(no_outliarsresident_df,logcurvefunction(no_outliarsresident_df,*popt),'r-',color ='orange')
m2,c2 =np.polyfit(no_outliarsresident_df,no_outliars_mean,1) #working out the parameters of the best fit line
print('m2',m2) #gradient
print("c2",c2) #intercept

#piecewise function
#Conditions: <4: straight line using m2 and c2, for 4<= use the logcurvefunction
#input is going to be the number of residents 1:7
p_residents = [1,2,3,4,5,6,7]
one = 1*m2+c2
two = 2*m2+c2
three = 3*m2+c2
four = logcurvefunction(4,*popt)
five = logcurvefunction(5,*popt)
six = logcurvefunction(6,*popt)
seven = logcurvefunction(7,*popt)
piecewise_values = [one,two,three,four,five,six,seven]
display(piecewise_values)
plt.figure(figsize = (4,3))
plotb = plt.plot(no_outliarsresident_df,no_outliars_mean,'o')
plt.xticks(no_outliarsresident_df,fontweight = 'bold')
plt.title("Number of Residents Vs. Average Yearly Energy Consumption (kWh) ",fontweight =  'bold',fontsize = 8)
plt.xlabel("Number of Residents",fontweight =  'bold',fontsize = 7)
plt.ylabel("Mean Yearly Energy Consumption (kWh)",fontweight =  'bold',fontsize = 7)
plt.plot(p_residents,piecewise_values,'r-',color ='orange')


# In[ ]:




