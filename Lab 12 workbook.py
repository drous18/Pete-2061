# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:36:34 2019

@author: Dillon Rousset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3

conn= sqlite3.connect("DCA.db")
cur = conn.cursor()

#cur.execute("ALTER TABLE Rates ADD rateID INTEGER;")
#conn.commit()

cur.execute("DROP TABLE DCAparams;")
cur.execute("DROP TABLE Rates;")
conn.commit()

titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13

cur.execute("CREATE TABLE DCAparams (wellID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,  qi REAL, Di REAL, b REAL, fluid TEXT)")
conn.commit()

dfLength = 24
gasWellID = np.random.randint(1,17,5)

for wellID in range(1,18):
    fileName = 'DCAwells_Solved/DCA_Well ' + str(wellID) + '.xlsx'
    
    xl = pd.ExcelFile(fileName)
    
    df1 = xl.parse('DCARegression')
    
    rateDF = pd.DataFrame({'wellID':wellID*np.ones(dfLength,dtype=int), 'time':range(1,dfLength+1),'rate':df1.iloc[8:32,1].values})
    rateDF['Cum'] = rateDF['rate'].cumsum()
    
    qi = df1.iloc[2,3]
    Di = df1.iloc[3,3]
    b  = df1.iloc[4,3]
    
    if wellID in gasWellID:
        cur.execute("INSERT INTO DCAparams VALUES ({},{},{},{},'gas')".format(wellID, qi, Di, b))
    else:
        cur.execute("INSERT INTO DCAparams VALUES ({},{},{},{},'oil')".format(wellID, qi, Di, b))

    conn.commit()
    
    t = np.arange(1,dfLength+1)
    Di = Di/12   
    
    if b>0:
        q = 30.4375*qi/((1 + b*Di*t)**(1/b))
        Np = 30.4375*(qi/(Di*(1-b)))*(1-(1/(1+(b*Di*t))**((1-b)/b))) #30.4375 = 365.125/12
    else:
        q = qi*np.exp(-Di*t)
        Np = 30.4375*(qi-q)/Di
        q = 30.4375*q
        
    error_q = rateDF['rate'].values - q
    SSE_q = np.dot(error_q, error_q)
    
    errorNp = rateDF['Cum'].values - Np
    SSE_Np = np.dot(errorNp,errorNp)
    
    
    rateDF['q_model'] = q
    rateDF['Cum_model'] = Np
    
    rateDF.to_sql("Rates", conn, if_exists="append", index = False)


    prodDF = pd.read_sql_query(f"SELECT time,Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)    
    dcaDF = pd.read_sql_query("SELECT * FROM DCAparams;", conn)
    
    currFig = plt.figure(figsize=(7,5), dpi=100)
    axes = currFig.add_axes([0.15, 0.15, 0.7, 0.7])
    
    axes.plot(prodDF['time'], prodDF['Cum']/1000, color="red", ls='None', marker='o', markersize=5,label = 'well '+str(wellID) )
    axes.plot(prodDF['time'], prodDF['Cum_model']/1000, color="red", lw=3, ls='-',label = 'well '+str(wellID) )
    axes.legend(loc=4)
    axes.set_title('Cumulative Production vs Time', fontsize=titleFontSize, fontweight='bold')
    axes.set_xlabel('Time, Months', fontsize=axisLabelFontSize, fontweight='bold') # Notice the use of set_ to begin methods
    axes.set_ylabel('Cumulative Production, Mbbls', fontsize=axisLabelFontSize, fontweight='bold')
    axes.set_ylim([0, 1200])
    axes.set_xlim([0, 25])
    xticks = range(0,30,5) 
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticks, fontsize=axisNumFontSize); 
    
    yticks = [0, 400, 800, 1200]
    axes.set_yticks(yticks)
    axes.set_yticklabels(yticks, fontsize=axisNumFontSize); 
    
    currFig.savefig('well'+str(wellID)+'_Gp.png', dpi=600)
    
cur.execute("ALTER TABLE Rates RENAME TO _old_Rates;")
cur.execute("CREATE TABLE Rates                                  \
(                                                                \
  rateID INTEGER PRIMARY KEY AUTOINCREMENT,                      \
  wellID INTEGER NOT NULL,                                       \
  time INTEGER NOT NULL,                                         \
  rate REAL, \
  Cum REAL, \
  q_model REAL, \
  Cum_model REAL, \
  CONSTRAINT fk_DCAparams                                        \
    FOREIGN KEY (wellID)                                         \
    REFERENCES DCAparams (wellID)                                \
);")

cur.execute("INSERT INTO Rates (wellID, time, rate, Cum, q_model, Cum_model) \
            SELECT wellID, time, rate, Cum, q_model, Cum_model FROM _old_Rates;")
conn.commit()

conn.close()

conn = sqlite3.connect("DCA.db")  #It will only connect to the DB if it already exists
#create data table to store summary info about each case/well
cur = conn.cursor()

wellID = 2
df2 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 7
df7 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 8
df8 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 16
df16 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
labels = [ "well 2", "well 7", "well 8", "well 16"]
fig, ax = plt.subplots()
ax.stackplot(df1['time'], df2['Cum']/1000, df7['Cum']/1000, df8['Cum']/1000, df16['Cum']/1000, labels=labels)
ax.legend(loc='upper left')
plt.title("Gas Well Stack Plot")
plt.show()

wellID = 1
df1 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 3
df3 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 
wellID = 4
df4 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 5
df5 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 6
df6 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 9
df9 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 10
df10 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 11
df11 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 12
df12 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 13
df13 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 14
df14 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 15
df15 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 17
df17 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)

labels = ["well 1", "well 3", "well 4", "well 5", "well 6", "well 9", "well 10", "well 11", "well 12", "well 13", "well 14", "well 15", "well 17"]
fig, ax = plt.subplots()
ax.stackplot(df1['time'], df1['Cum']/1000, df3['Cum']/1000, df4['Cum']/1000, df5['Cum']/1000, df6['Cum']/1000, df9['Cum']/1000, df10['Cum']/1000, df11['Cum']/1000, df12['Cum']/1000, df13['Cum']/1000, df14['Cum']/1000, df15['Cum']/1000, df17['Cum']/1000, labels=labels)
ax.legend(loc='upper left')
plt.title("Oil Well Stack Plot")
plt.show()



#stacked bar graph
N = 6
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun']
width = 0.5       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(df2['time'][0:N], df2['Cum'][0:N]/1000, width)
p2 = plt.bar(df2['time'][0:N], df7['Cum'][0:N]/1000, width, bottom=df2['Cum'][0:N]/1000)
p3 = plt.bar(df2['time'][0:N], df8['Cum'][0:N]/1000, width, bottom=(df2['Cum'][0:N]+df7['Cum'][0:N])/1000)
p4 = plt.bar(df2['time'][0:N], df16['Cum'][0:N]/1000, width, bottom=(df2['Cum'][0:N]+df7['Cum'][0:N]+df8['Cum'][0:N])/1000)

plt.ylabel('Gas Production, Mbbls')
plt.title('Cumulative Production Forecast')
plt.xticks(ind, months, fontweight='bold')
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('well 2', 'well 7', 'well 8', 'well 16'))
plt.title("Gas Well Bar Graph")
plt.show()

N = 6
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun']
width = 0.5       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(df1['time'][0:N], df1['Cum'][0:N]/1000, width)
p2 = plt.bar(df1['time'][0:N], df3['Cum'][0:N]/1000, width, bottom=df1['Cum'][0:N]/1000)
p3 = plt.bar(df1['time'][0:N], df4['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N])/1000)
p4 = plt.bar(df1['time'][0:N], df5['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N])/1000)
p5 = plt.bar(df1['time'][0:N], df6['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N])/1000)
p6 = plt.bar(df1['time'][0:N], df9['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N])/1000)
p7 = plt.bar(df1['time'][0:N], df10['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N])/1000)
p8 = plt.bar(df1['time'][0:N], df11['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N])/1000)
p9 = plt.bar(df1['time'][0:N], df12['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N]+df11['Cum'][0:N])/1000)
p10 = plt.bar(df1['time'][0:N], df13['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N])/1000)
p11 = plt.bar(df1['time'][0:N], df14['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N])/1000)
p12 = plt.bar(df1['time'][0:N], df15['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N]+df14['Cum'][0:N])/1000)
p13 = plt.bar(df1['time'][0:N], df17['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df3['Cum'][0:N]+df4['Cum'][0:N]+df5['Cum'][0:N]+df6['Cum'][0:N]+df9['Cum'][0:N]+df10['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N]+df14['Cum'][0:N]+df15['Cum'][0:N])/1000)

plt.ylabel('Oil Production, Mbbls')
plt.title('Cumulative Production Forecast')
plt.xticks(ind, months, fontweight='bold')
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0]), ("well 1", "well 3", "well 4", "well 5", "well 6", "well 9", "well 10", "well 11", "well 12", "well 13", "well 14", "well 15", "well 17"))
plt.title("Oil Well Bar Graph")
plt.show()

# Primary and Secondary Y-axes
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df1['time'], df1['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df1['time'], df1['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 1")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df2['time'], df2['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df2['time'], df2['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 2")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df3['time'], df3['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df3['time'], df3['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 3")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df4['time'], df4['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df4['time'], df4['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 4")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df5['time'], df5['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df5['time'], df5['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 5")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df6['time'], df6['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df6['time'], df6['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 6")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df7['time'], df7['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df7['time'], df7['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 7")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df8['time'], df8['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df8['time'], df8['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 8")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df9['time'], df9['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df9['time'], df9['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 9")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df10['time'], df10['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df10['time'], df10['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 10")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df11['time'], df11['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df11['time'], df11['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 11")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df12['time'], df12['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df12['time'], df12['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 12")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df13['time'], df13['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df13['time'], df13['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 13")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df14['time'], df14['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df14['time'], df14['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 14")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df15['time'], df15['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df15['time'], df15['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 15")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df16['time'], df16['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df16['time'], df16['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 16")
plt.show()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df17['time'], df17['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df17['time'], df17['Cum']/1000, 'g-')

ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.title("Well 17")
plt.show()




#two approaches to load a LAS input file
data1 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)
data1DF = pd.read_csv("volve_logs/15_9-F-1B_INPUT.LAS",skiprows=69, sep = '\s+' )


data = np.loadtxt("WLC_PETRO_COMPUTED_INPUT_1.DLIS.0.las", skiprows=48)
DZ,rho=data[:,0], data[:,1]


DZ=DZ[np.where(rho>0)]
rho=rho[np.where(rho>0)]

print('Investigated Depth',[min(DZ),max(DZ)])

fig = plt.figure(figsize=(5,15), dpi=100)
plt.plot(rho,DZ, color='blue')
plt.xlabel('Density, g/cc', fontsize = 14, fontweight='bold')
plt.ylabel('Depth, m', fontsize = 14, fontweight='bold')
plt.gca().invert_yaxis()
plt.show()

titleFontSize = 22
fontSize = 20

fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho,DZ, color='red')
plt.plot(rho*1.1,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(rho,DZ, color='green')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(rho,DZ, color='blue')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(rho,DZ, color='black')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(rho,DZ, color='brown')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(rho,DZ, color='grey')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('well_1_log.png', dpi=600)