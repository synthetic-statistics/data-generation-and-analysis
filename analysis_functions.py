# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import factorial

#%%
def make_filters_from_nominal_variable(df,column_name):
    unique_names=np.unique(df[column_name])
    for i in unique_names:
        df[str(i)+"_filter"]=df[column_name]==i


#%%
def split_dataset(df,timestep_threshold,startname="dynamic",endname="static"):
    """ create two filters that split the dataframe df, 
    in one dataset before the timestep_threshold and one after (including the threshold value)
    """
    threshold=np.where(df["time_step"]>=timestep_threshold)[0][0]
    bool_var=np.zeros(df["time_step"].shape,dtype=bool)
    bool_var[threshold:]=True
    df[endname]=bool_var
    bool_var2=np.invert(bool_var)
    df[startname]=bool_var2

#%%
def descriptive_stats(df,column_name,filter_bool=None):#,personality=None
    """get time dependent results for the quartiles (as first return variable)
    and the time dependent  mean and standarad deviation (as second return variable)  
    """
    timestep=np.arange(np.max(df["time_step"]+1))
    
    order_stat=np.empty([len(timestep),5])
    normal_stat=np.empty([len(timestep),2])

    
    for i in range(len(timestep)): 
        bool_filter=df["time_step"]==timestep[i]

        if filter_bool is not None:
            if filter_bool.dtype == bool:
                bool_filter *= filter_bool
            else:
                bool_filter *= np.invert(np.isnan(filter_bool))

        data=df[column_name][bool_filter]

        order=np.quantile(data,[0,.25,.5,.75,1])
        order_stat[i]=order
        normal_stat[i,0]=np.mean(data)
        normal_stat[i,1]=np.std(data)
    return order_stat,normal_stat    

#%%
def distribution_at_timesteps(df,column_name,res=60,filter_bool=None):
    """create a 2D-array from the time dependent histograms of the specified variable
    filtering can be done via the argument filter_bool 
    """
    timestep=np.arange(np.max(df["time_step"]+1))

    if filter_bool is None:
        filter_bool=np.ones(df[column_name].shape,dtype=bool)
    elif filter_bool.dtype != bool:
        filter_bool = np.invert(np.isnan(filter_bool))
        
    data=df[column_name][filter_bool]
    global_min=np.min(data)
    global_max=np.max(data)
        
    bins=np.linspace(global_min,global_max,res+1)
    bin_centers=bins[:-1]+np.diff(bins)/2
    
    result=np.empty([len(timestep),res])
    for i in range(len(timestep)): 
        time_filter=df["time_step"]==timestep[i]
        data=df[column_name][filter_bool*time_filter]
        vals,bins=np.histogram(data,bins)
        result[i]=vals

    return result,bin_centers  

#%%
def plot_descr_and_distr(df,column_name,res=60,aspect=0.5,filter_bool=None):
    fig,axs=plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(16)
    order,normal=descriptive_stats(df,column_name,filter_bool=filter_bool)    
    alpha=[0.3,0.6,1,0.6,0.3]
    labels=["min ; max","25% ; 75%","median"]
    for i in range(5):
        if i<3:
            axs[0].plot(order[:,i],c='b',alpha=alpha[i],label=labels[i])
        else:
            axs[0].plot(order[:,i],c='b',alpha=alpha[i])
    
    axs[0].plot(normal[:,0],c='r',label='mean')
    axs[0].plot(normal[:,0]-normal[:,1],'--',c='r',label=r'$\pm$ std')
    axs[0].plot(normal[:,0]+normal[:,1],'--',c='r')
    
    axs[0].set_xlabel("time step")
    axs[0].set_ylabel(column_name)
    axs[0].legend()

    img,bc=distribution_at_timesteps(df,column_name,res=res,filter_bool=filter_bool)
    
    ytick_every=int((res-1)/6)
    if ytick_every<1:
        ytick_every=1

    surf=axs[1].imshow(img[:,::-1].T,cmap='plasma',aspect=aspect)
    axs[1].set_xlabel("time step")
    axs[1].set_ylabel(column_name)
    
    yticks=np.arange(img.shape[1])[::-1][::ytick_every]
    yticklabs=np.round(bc[::ytick_every],2)
    axs[1].set(yticks=yticks, yticklabels=yticklabs)
    fig.colorbar(surf, shrink=1., aspect=12,label='frequency')
    #plt.show()
    return fig,axs

#%%
def color_scatterplot(df,x_name,y_name,c_name,cmap_name="plasma",alpha=1,clims=None,filter_bool=None,figsize=None):
    x=df[x_name]
    y=df[y_name]
    c=df[c_name]
    
    bool_filter=np.invert(np.isnan(x))*np.invert(np.isnan(y))*np.invert(np.isnan(c))
    if filter_bool is not None:
        if filter_bool.dtype == bool:
            bool_filter *= filter_bool
        else:
            bool_filter *= np.invert(np.isnan(filter_bool))
    
    xdat=x[bool_filter]
    ydat=y[bool_filter]
    cdat=c[bool_filter]

    print("number of points: "+str(len(cdat)))
    if clims is None:
        mincdat=np.min(cdat)
        maxcdat=np.max(cdat)
        cdat_normed=cdat-mincdat
        cdat_normed /= np.max(cdat_normed)
    else:
        mincdat=clims[0]
        maxcdat=clims[1]
        cdat_normed=cdat-mincdat
        cdat_normed /= (maxcdat-mincdat)
    

    fig,ax=plt.subplots()
    if figsize is not None:
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[0])
    cmap = mpl.colormaps[cmap_name]
    colors = cmap(cdat_normed,alpha=alpha)
    scatter=ax.scatter(xdat,ydat,c=colors)
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(mincdat, maxcdat), cmap=cmap_name),
             ax=ax, orientation='vertical', label=c_name)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    return fig,ax

#%%

def group_shares_to_unique_string(df):
    """create group_code  n o c : normal overachiever charismatic_idiot
    """
    h1=df["group_share_normal"]*5*100
    h2=df["group_share_overachiever"]*5*10
    h3=df["group_share_charismatic_idiot"]*5
    group_code=(h1+h2+h3).astype(int)
    group_code=group_code.astype(str)
    for i in range(len(group_code)):
        gc=group_code[i].zfill(3)
        #group_code[i]="n:"+gc[0]+"-o:"+gc[1]+"-c:"+gc[2]  #more explicit
        group_code[i]=gc[0]+"-"+gc[1]+"-"+gc[2]  #more compact
    df["group_composition_n-o-c"]=group_code

#%%
def create_single_point_per_group_filter(df,group_size=5):
    newfilter=np.zeros(df["group_id"].shape,dtype=bool)
    indices=np.arange(0,len(df["group_id"]),group_size)
    newfilter[indices]=True
    df["single_point_per_group_filter"]=newfilter

#%%

def datachaosrepr_linear(r,n,path):
    a=2*np.pi/(n*1.)
    x=[]
    y=[]
    for i in range(n):
        x.append(np.sin(a*i+0.25*np.pi))
        y.append(np.cos(a*i+0.25*np.pi))
    x1=[]
    y1=[]
    ystart=0#5
    xstart=0#5
    while xstart**2+ystart**2 > 1:
        xstart=np.random.uniform(-1,1)
        ystart=np.random.uniform(-1,1)
    y1.append(ystart)
    x1.append(xstart)
    for i in range(len(path)):
        x1.append(x1[i-1]+r*x[int(path[i])])
        y1.append(y1[i-1]+r*y[int(path[i])])
    return x1,y1,x,y

#%%
def group_composition_code_to_index(ugt_array,unique_grouptypes):
    #unique_gtypes=np.unique(ugt_array)
    N=len(unique_grouptypes)
    
    indices=np.arange(N)
    converter_dict=dict()
    for c,i in enumerate(unique_grouptypes):
        converter_dict[i]=c
        
    new=np.empty(ugt_array.shape,dtype=int)
    for i in range(len(ugt_array)):
        new[i]=converter_dict[ugt_array[i]]
    back_conversion=dict()
    for key,val in converter_dict.items():
        back_conversion[val]=key
    return new,back_conversion


#%%

def datachaosrepr(r,n,path):
    a=2*np.pi/(n*1.)
    x=[]
    y=[]
    for i in range(n):
        x.append(np.sin(a*i+0.25*np.pi))
        y.append(np.cos(a*i+0.25*np.pi))
    x1=[]
    y1=[]
    ystart=5
    xstart=5
    while xstart**2+ystart**2 > 1:
        xstart=np.random.uniform(-1,1)
        ystart=np.random.uniform(-1,1)
    y1.append(ystart)#0
    x1.append(xstart)#0
    for i in range(len(path)):
        x1.append(x1[i-1]+r*(x[int(path[i])]-x1[i-1]))
        y1.append(y1[i-1]+r*(y[int(path[i])]-y1[i-1]))
    return x1,y1,x,y

#%%

def initial_probabilities(population_distribution,group_size):#all_group_types):
    #group_size=int(group_size)
    all_grouptypes=[]
    for i in range(group_size+1):
        for j in range(group_size+1):
            for k in range(group_size+1):
                if i+j+k == group_size:
                    all_grouptypes.append(str(i)+"-"+str(j)+"-"+str(k))
    
    initial_prob=[]
    nfactor=factorial(group_size)
    for group_code in all_grouptypes:    
        number_list=np.array(group_code.split("-")).astype(int)
        prob=1
        factor=1
        for j in range(len(number_list)):
            prob *= population_distribution[j]**number_list[j]
            factor *= factorial(number_list[j])
        prob*=nfactor/factor
        initial_prob.append(prob)
    #normalization=np.sum(initial_prob)
    #print(normalization)# is 1 
    return all_grouptypes,np.array(initial_prob)#/normalization

#%%

# time since last win
def create_time_since_last_win_col(df,group_size=5):
    number_of_groups=len(np.unique(df["group_id"]))
    lastwins=np.zeros(number_of_groups)+np.nan
    wincounters=np.zeros(number_of_groups)
    
    number_of_rows=len(df["group_id"])
    new_col=np.empty(number_of_rows)
    for i in range(number_of_rows):
        timestep=df["time_step"][i]
        group=df["group_id"][i]

        if np.isnan(lastwins[group]):
            new_col[i]=np.nan
        else:
            new_col[i]=timestep-lastwins[group]    
        if df["group_winning"][i]:
            wincounters[group]+=1
            if wincounters[group]==5:
                wincounters[group]=0
                lastwins[group]=timestep
    df["time_since_last_win"]=new_col

#%%
def group_composition_occurence_frequency(df,all_grouptypes,filter_bool=None):
    """counting the occurences of each group type
    (counted, when the group composition appears,
    not double counting a non-changing composition over time) 
    """
    if filter_bool is None:
        filter_bool=np.ones(df[column_name].shape,dtype=bool)
    elif filter_bool.dtype != bool:
        filter_bool = np.invert(np.isnan(filter_bool))

    
    filter1=np.bitwise_or(df["time_since_last_win"]==1,df["time_step"]==0)
    filter2=filter1*df["single_point_per_group_filter"]

    filtered_data=df["group_composition_n-o-c"][filter2*filter_bool]
    freqs=[]
    for i in all_grouptypes:
        freq=np.sum(filtered_data==i)
        freqs.append(freq)
    return freqs
#%%

def create_group_score_increased_by_learning(df,group_size=5,number_of_groups=200):
    new=np.zeros(df["time_step"].shape)+np.nan    
    filter1=(df["time_since_last_win"]!=1) * (df["time_step"]>0)# * np.invert(np.isnan(df["time_since_last_win"]))
    index_timestep=int(group_size*number_of_groups)
    for i in range(len(df["time_step"])):
        if filter1[i]:
            new[i]=df["group_score"][i]-df["group_score"][i-index_timestep]
    df["group_score_increased_by_learning"]=new
#%%

def create_performance_increased_by_learning(df,group_size=5,number_of_groups=200):
    new=np.zeros(df["time_step"].shape)+np.nan    
    filter1=(df["time_since_last_win"]!=1) * (df["time_step"]>0)#* np.invert(np.isnan(df["time_since_last_win"]))
    index_timestep=int(group_size*number_of_groups)
    for i in range(len(df["time_step"])):
        if filter1[i]:
            new[i]=df["performance"][i]-df["performance"][i-index_timestep]
    df["performance_increased_by_learning"]=new
    
#%%
def performance_change_after_rehiring(df,group_size=5,number_of_groups=200):
    new=np.zeros(df["time_step"].shape)+np.nan    
    filter1=df["time_since_last_win"]==1
    index_timestep=int(group_size*number_of_groups)
    for i in range(len(df["time_step"])):
        if filter1[i]:
            new[i]=df["performance"][i]-df["performance"][i-index_timestep]
    df["performance_change_after_rehiring"]=new

#%%
def group_score_change_after_rehiring(df,group_size=5,number_of_groups=200):
    new=np.zeros(df["time_step"].shape)+np.nan    
    filter1=df["time_since_last_win"]==1
    index_timestep=int(group_size*number_of_groups)
    for i in range(len(df["time_step"])):
        if filter1[i]:
            new[i]=df["group_score"][i]-df["group_score"][i-index_timestep]
    df["group_score_change_after_rehiring"]=new
#%%
def newest_group_member_and_last_promoted_personality(df,group_size=5,number_of_groups=200):
    new=["NA" for i in range(len(df["time_step"]))]
    prom=["NA" for i in range(len(df["time_step"]))]
    new_guy_filter=np.zeros(len(df["time_step"]),dtype=bool)
    filter1=df["time_since_last_win"]==1
    timesteps=np.arange(np.max(df["time_step"]))+1
    index_timestep=int(group_size*number_of_groups)
    index_offset=0
    for i in range(number_of_groups):
        newest="NA"
        last_promoted="NA"
        k_index=None
        for j in timesteps:
            t_index=index_offset+j*index_timestep
            if filter1[t_index]:
                for k in range(group_size):
                    if not np.isnan(df["promotion_time"][t_index-index_timestep+k]):
                        last_promoted=df["personality"][t_index-index_timestep+k]
                        k_index=k
                newest=df["personality"][t_index+k_index]    
                
            if k_index is not None:
                new_guy_filter[t_index+k_index]=True
            
            for k in range(group_size):
                new[t_index+k]=newest
                prom[t_index+k]=last_promoted

        index_offset+=group_size

    df["new_guy_filter"]=new_guy_filter
    df["last_promoted"]=prom
    df["newest_group_member"]=new

#%%
def intra_group_stats(df,group_size=5): # (inverse: group homogeneity)
    giprfm=np.zeros(df["group_score"].shape)
    for i in range(len(df["group_score"])):
        giprfm[i]=df["performance"][i]-df["group_score"][i]/group_size

    df["group_internal_performance_residiual_from_mean"]=giprfm

    counter=0
    helper=0
    indices=[]
    igp2=giprfm**2
    gis=np.zeros(df["group_score"].shape)
    for i in range(len(df["group_score"])):
        counter+=1
        indices.append(i)
        helper+=igp2[i]
        if counter==group_size:
            counter=0
            helper=np.sqrt(helper/group_size)
            gis[indices]=helper
            helper=0
            indices=[]

    df["group_internal_performance_std"]=gis

#%%

def filter_consecutive_repetitions(sequence):
    new_filter=np.zeros(sequence.shape,dtype=bool)
    new_filter[0]=True
    for i in range(1,len(sequence)):
        if sequence[i]!= sequence[i-1]:
            new_filter[i]=True
    return new_filter




    