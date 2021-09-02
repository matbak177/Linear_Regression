import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simpledbf import Dbf5
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import os
%matplotlib inline

#change path
folder=r'C:\Users\mateusz.bak\Desktop\programs\regression'
daty = [x for x in os.listdir(folder) if x.startswith('2020')]
 
# load file  
def load(path):
    file=Dbf5(path)
    file=file.to_dataframe()
    file_c=file.copy()
    return file_c

# Creating a regression line
def regression(frame,col):
    X=frame["C_1"].values
    y=frame[col].values

    df=pd.DataFrame({'date':X,'val':y})
    df.date=pd.to_datetime(df.date)

    lr=linear_model.LinearRegression()
    lr.fit(df.date.values.reshape(-1,1),df['val'])#.reshape(-1,1)

    Y_pred=lr.predict(df.date.values.astype(float).reshape(-1,1))
    df['pred']=Y_pred

    return df
    #ax = df.plot(x='date', y='val', color='black', style='.')
    #df.plot(x='date', y='pred', color='orange', linewidth=3, ax=ax, alpha=0.5)

# creating charts
def plots(frame,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,title):

    f=plt.figure()
    f, axes = plt.subplots(nrows=3,ncols=4,figsize=(30,20))
    
    axes[0][0].plot(df1['date'],df1['pred'],color='red',linewidth=3)
    axes[0][0].scatter(x=frame['C_1'],y=frame['C_2'])
    axes[0][0].title.set_text("B1")
    
    axes[0][1].plot(df2['date'],df2['pred'],color='red',linewidth=3)
    axes[0][1].scatter(x=frame['C_1'],y=frame['C_3'])
    axes[0][1].title.set_text("B2")

    axes[0][2].plot(df3['date'],df3['pred'],color='red',linewidth=3)    
    axes[0][2].scatter(x=frame['C_1'],y=frame['C_4'])
    axes[0][2].title.set_text("B3")
    
    axes[0][3].plot(df4['date'],df4['pred'],color='red',linewidth=3)    
    axes[0][3].scatter(x=frame['C_1'],y=frame['C_5'])
    axes[0][3].title.set_text("B4")
    
    axes[1][0].plot(df5['date'],df5['pred'],color='red',linewidth=3)   
    axes[1][0].scatter(x=frame['C_1'],y=frame['C_6'])
    axes[1][0].title.set_text("B5")

    axes[1][1].plot(df6['date'],df6['pred'],color='red',linewidth=3)    
    axes[1][1].scatter(x=frame['C_1'],y=frame['C_7'])
    axes[1][1].title.set_text("B6")

    axes[1][2].plot(df7['date'],df7['pred'],color='red',linewidth=3)    
    axes[1][2].scatter(x=frame['C_1'],y=frame['C_8'])
    axes[1][2].title.set_text("B7")

    axes[1][3].plot(df8['date'],df8['pred'],color='red',linewidth=3)    
    axes[1][3].scatter(x=frame['C_1'],y=frame['C_9'])
    axes[1][3].title.set_text("B8")
    
    axes[2][0].plot(df9['date'],df9['pred'],color='red',linewidth=3)    
    axes[2][0].scatter(x=frame['C_1'],y=frame['C_10'])
    axes[2][0].title.set_text("B9")

    axes[2][1].plot(df10['date'],df10['pred'],color='red',linewidth=3)    
    axes[2][1].scatter(x=frame['C_1'],y=frame['C_11'])
    axes[2][1].title.set_text("B10")

    axes[2][2].plot(df11['date'],df11['pred'],color='red',linewidth=3)    
    axes[2][2].scatter(x=frame['C_1'],y=frame['C_12'])
    axes[2][2].title.set_text("B11")

    axes[2][3].plot(df12['date'],df12['pred'],color='red',linewidth=3)    
    axes[2][3].scatter(x=frame['C_1'],y=frame['C_13'])
    axes[2][3].title.set_text("B12")
    
    f.suptitle(title,fontsize=50)
    plt.show()

nazwy=['trw','ws','wp','zab','igl','lisc']
columns=['C_2','C_3','C_4','C_5','C_6','C_7','C_8','C_9','C_10','C_11','C_12','C_13']
  
# load all files 
q=0
for d in daty:
    q+=1
    folder2=os.path.join(folder,d)
    files=[x for x in os.listdir(folder2) if x.endswith('p.dbf')]

    for o,p in zip(files,nazwy):
        path=(os.path.join(folder,d,o))
        exec('{}=load(path)'.format(p+str(q)))
        exec('{}["CID"]="{}"'.format(p+str(q),d))
        exec('{}.columns=["C_"+str(i) for i in range(1, 14)]'.format(p+str(q)))
        
#joining DataFrames
trw=pd.concat([trw1,trw2,trw3,trw4,trw5,trw6,trw7,trw8],ignore_index=True)
igl=pd.concat([igl1,igl2,igl3,igl4,igl5,igl6,igl7,igl8],ignore_index=True)
lisc=pd.concat([lisc1,lisc2,lisc3,lisc4,lisc5,lisc6,lisc7,lisc8],ignore_index=True)
wp=pd.concat([wp1,wp2,wp3,wp4,wp5,wp6,wp7,wp8],ignore_index=True)
ws=pd.concat([ws1,ws2,ws3,ws4,ws5,ws6,ws7,ws8],ignore_index=True)
zab=pd.concat([zab1,zab2,zab3,zab4,zab5,zab6,zab7,zab8],ignore_index=True)

# creating a line regression for columns of variables
con=[trw,ws,wp,zab,igl,lisc]
for c,n in zip(con,nazwy):
    v=1
    dfs=[]
    for r in columns:
        exec('{}_df_{}=regression(c,r)'.format(n,v))
        dfs.append('{}_df_{}'.format(n,v))
        v+=1

# creating graphs from variables    
plots(trw,trw_df_1,trw_df_2,trw_df_3,trw_df_4,trw_df_5,trw_df_6,trw_df_7
      ,trw_df_8,trw_df_9,trw_df_10,trw_df_11,trw_df_12,'GRASS')
        
plots(wp,wp_df_1,wp_df_2,wp_df_3,wp_df_4,wp_df_5,wp_df_6,wp_df_7,
      wp_df_8,wp_df_9,wp_df_10,wp_df_11,wp_df_12,'FLOWING WATER')

plots(ws,ws_df_1,ws_df_2,ws_df_3,ws_df_4,ws_df_5,ws_df_6,ws_df_7,
      ws_df_8,ws_df_9,ws_df_10,ws_df_11,ws_df_12,'STAGNANT WATER')

plots(zab,zab_df_1,zab_df_2,zab_df_3,zab_df_4,zab_df_5,zab_df_6,zab_df_7,
      zab_df_8,zab_df_9,zab_df_10,zab_df_11,zab_df_12,'BUILDING')

plots(lisc,lisc_df_1,lisc_df_2,lisc_df_3,lisc_df_4,lisc_df_5,lisc_df_6,lisc_df_7,
      lisc_df_8,lisc_df_9,lisc_df_10,lisc_df_11,lisc_df_12,'DECIDUOUS FOREST') 

plots(igl,igl_df_1,igl_df_2,igl_df_3,igl_df_4,igl_df_5,igl_df_6,igl_df_7,
      igl_df_8,igl_df_9,igl_df_10,igl_df_11,igl_df_12,'CONIFEROUS FOREST')
    





