# -*- coding: utf-8 -*-

import numpy as np
import os
import random

def folder_file(path_string):
    """
    split a string presenting a path with the filename into the filename and the 
    directory-path. (works with slash, backslash or double backslash as seperator)

    Args:
        path_string (string): 
            absolute or relative path.

    Returns:
        directory_path (string): 
            folder.
        
        filename (string): 
            file.

    """
    
    name = path_string.replace("\\", "/")
    pos = name[::-1].find("/")
    return name[:-pos], name[-pos:]

folder,file=folder_file(__file__)
os.chdir(folder)

import data_generation_functions as dgf

#%% initialize the log text-file
logger=dgf.data_generation_logger.create('synthetic.log')

logger.info('##################################')
logger.info('Synthetic flirt data: Creation Log')
logger.info('##################################')

logger.info("")


#%%
names=dict()

names["physical attractiveness"]=[] ######

names["physical attractiveness"].append("sportive activities")
names["physical attractiveness"].append("age")
names["physical attractiveness"].append("bmi")
names["physical attractiveness"].append("style / design")
names["physical attractiveness"].append("body scent")
names["physical attractiveness"].append("facial symmetry")


names["social status"]=[]  ######

names["social status"].append("wealth")
names["social status"].append("educational background")
names["social status"].append("family background")
names["social status"].append("income")
names["social status"].append("occupational prestige")
names["social status"].append("health")


names["social intelligence"]=[] ######

names["social intelligence"].append("empathy")
names["social intelligence"].append("self awareness")
names["social intelligence"].append("active listening")
names["social intelligence"].append("social / situational awareness")
names["social intelligence"].append("humour")
names["social intelligence"].append("emotional expressiveness")

names["self confidence"]=[] ######

names["self confidence"].append("self acceptance")
names["self confidence"].append("stress tolerance")
names["self confidence"].append("risk-taking willingness")
names["self confidence"].append("humility")
names["self confidence"].append("optimism")
names["self confidence"].append("trustworthiness / responsiblity")


names["rhetoric abilities / charisma"]=[]  ######

names["rhetoric abilities / charisma"].append("eloquence / language usage")
names["rhetoric abilities / charisma"].append("self control")
names["rhetoric abilities / charisma"].append("body language")
names["rhetoric abilities / charisma"].append("assertiveness")
names["rhetoric abilities / charisma"].append("voice")
names["rhetoric abilities / charisma"].append("enthusiasm")



#%%
popsize=10000
scales=5
ratio=1/scales
subscales=6
epsilons=[0.5,1,2,4,8]


beta=np.random.uniform(-2,2,18)
beta[1]=np.abs(beta[1])
beta[3]=np.abs(beta[3])

logger.info("all input parameters (betas and for variables mu and sigma)")
logger.info(str(beta))
logger.info("")
logger.info("all input noise levels for epsilons mu and sigma")
logger.info(str(epsilons))
logger.info("")

logger.info("")
logger.info("detailed description")
logger.info("")


subscaledata=[]
scale_means=np.empty([popsize,scales])

nameskeylist=list(names.keys())
for i in range(scales):
    logger.info("")
    logger.info(nameskeylist[i])
    logger.info("-------------------------------------------------------------------------------------")
    logger.info("")
    
    epsilon=epsilons[i]

    logger.info(names[nameskeylist[i]][0])
    a=dgf.normal_dist(beta[0],beta[1],popsize)
    az=dgf.z_transform(a)
    
    logger.info(names[nameskeylist[i]][1])
    b=dgf.normal_dist(beta[2],beta[3],popsize)
    bz=dgf.z_transform(b)
    
    logger.info(names[nameskeylist[i]][2])
    logger.info("dependent on variables:")
    logger.info("    X1 = "+names[nameskeylist[i]][0])
    logger.info("    X2 = "+names[nameskeylist[i]][1])
    c=dgf.gen_correlate([az,bz], beta[4], beta[5:7], epsilon)
    cz=dgf.z_transform(c)

    logger.info(names[nameskeylist[i]][3])
    logger.info("dependent on variables:")
    logger.info("    X1 = "+names[nameskeylist[i]][0])
    logger.info("    X2 = "+names[nameskeylist[i]][1])
    d=dgf.gen_moderate(az, bz, beta[7], beta[8], beta[9], beta[10], epsilon)
    dz=dgf.z_transform(d)

    logger.info(names[nameskeylist[i]][4])
    logger.info("dependent on variables:")
    logger.info("    X1 = "+names[nameskeylist[i]][1])
    logger.info("    X2 = "+names[nameskeylist[i]][2])
    e=dgf.gen_correlate([bz,cz], beta[11], beta[12:14], epsilon)
    ez=dgf.z_transform(e)
    
    logger.info(names[nameskeylist[i]][5])
    logger.info("dependent on variables:")
    logger.info("    X1 = "+names[nameskeylist[i]][0])
    logger.info("    X2 = "+names[nameskeylist[i]][4])
    f=dgf.gen_moderate(az, ez, beta[14], beta[15], beta[16], beta[17], epsilon)
    fz=dgf.z_transform(f)
    scale_mean=(az+bz+cz+dz+ez+fz)/subscales
 
    scale_means[:,i]=scale_mean
    subscaledata.append(np.array((a,b,c,d,e,f)).T)
    
props=scale_means

#%% initialize groups
groups=dict()

for i in range(popsize):
    groups[i]=dict()
    groups[i]["members"]=[i]

#%% merge groups
shift=1
first=True

groupkeylist=np.array(list(groups.keys()))

counter=0
while True:
    counter+=1
    if counter%100==0:
        print(len(groups))
        print(shift)
        
    if len(groups)==1:
        break
    

    proposed_merge=dgf.propose_group_merges(groupkeylist,shift,first)

    groups,number_of_merges=dgf.check_and_execute_merge(proposed_merge, groups, props,ratio)
    
    if number_of_merges==0:
        if not first:
            shift+=1
        
        first=not first
        
        if shift>len(groups)//2:
            if len(groups)%2==0:
                break
            else:
                if not first:
                    break
    else:
        shift=1
        first=True
        groupkeylist=list(groups.keys())
        shuffled_list=random.sample(groupkeylist,len(groupkeylist))
        groupkeylist=np.array(shuffled_list)
        

#%% 
groupnumbers=np.empty(popsize,dtype=int)
for c,i in enumerate(groups):
    for j in groups[i]["members"]:
        groupnumbers[j]=c

#%% get all information into one big table (result)

result=np.empty([popsize,scales*subscales+scales+1])

result[:,0]=groupnumbers

result[:,1]=scale_means[:,0]
result[:,2:8]=subscaledata[0]

result[:,8]=scale_means[:,1]
result[:,9:15]=subscaledata[1]

result[:,15]=scale_means[:,2]
result[:,16:22]=subscaledata[2]

result[:,22]=scale_means[:,3]
result[:,23:29]=subscaledata[3]

result[:,29]=scale_means[:,4]
result[:,30:36]=subscaledata[4]


#%% create the header naming the columns of the table

header="clique"
delimiter=" , "

for i in names:
    header+= delimiter+i
    for j in names[i]:
        header+= delimiter + j

#%%
np.savetxt("synthetic_flirt_data.csv",result,header=header,delimiter=",")

#%%
dgf.data_generation_logger.close()
