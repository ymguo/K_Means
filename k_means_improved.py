#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:18:42 2019

@author: ymguo
"""

'''
Initialized seeds/centroids  :  K-Means++
[2007, Arthur & Vassilvitskii]
   
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import cv2

'''
K-Means++ : 初始化质心
    优化 初始点选择
'''

'''方法一'''
'''
def init_centroids(df,k):
    
    #step 1:随机寻找第一个质心
    centroids = {0: df.loc[np.random.randint(0, len(df))]}

    while k>1:
        for i in centroids.keys():
            df[i] = ((df['x'] - centroids[i][0]) ** 2+ (df['y'] - centroids[i][1]) ** 2)
            # 计算样本点与已存在质心的最小距离
            df['dist'] = df.iloc[:,2:len(centroids.keys())+2 ].min(axis=1)
            #step 2: 根据最小距离计算概率
            df['p'] = df['dist']/sum(df['dist'])
        
        #step 3：根据概率选择一个新的质心并加入质心列表
        a = np.random.rand()

        for indexs in df.index:
            a -= df.loc[indexs].values[-1]
            if a <0:
                b = {len(centroids):df.iloc[indexs,0:2]}               
                centroids.update(b)
                break;
        k-=1
        
    return centroids
'''

'''方法二'''
'''
#对初始选点进行优化
def kmeansplus(df, k):

    samplen=len(df['x'])

    #随机选择第一个中心点
    first = np.random.randint(0, samplen)
    centroids = {
        i: [np.zeros(2)]
        for i in range(k)
    }
    centroids[0][:] = df.loc[first, :]

    #选取后续的中心点
    dist = np.zeros(samplen)
    cen = 0
    for i in range(k-1):
        cen = cen + 1
        for j in range(cen):
            dist =dist + np.sqrt((df['x'] - centroids[j][0]) ** 2 + (df['y'] - centroids[j][1]) ** 2)
        rand = np.random.random() * sum(dist)
        # np.random.random() 生成随机浮点数，取值范围：[0,1)        
        sum_dist = 0
        for l in range(samplen):
            sum_dist = sum_dist + dist[l]
            if sum_dist >= rand:
                centroids[i+1][ :] = df.loc[l, :]
                break
    return centroids
'''

'''方法三 : slides in class'''
def select_center(first_center, df, k):
    center = {}
    center[0]=first_center
    D_x_2 = (df["x"]-first_center[0])**2+(df["y"]-first_center[1])**2
    P_x = D_x_2/(D_x_2.sum())
    sum_x = P_x.cumsum()
    for i in range(1,k):
        next_center = np.random.random()
        for index,j in enumerate(sum_x):
            if j>next_center:
                break
        center[i]=list(df.iloc[index].values)
    return center

'''
下面同 k-means
'''

def assignment(df, centroids, colmap):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )  # python mode : 没有一个一个sample去写
        )
    # color
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update(df, centroids):
    # recalculate the centroids
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def main():
    # step 0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    # dataframe 返回一个二维矩阵，
    # 用.loc直接定位
    
    k = 3
  
    # 以K-Means++算法 初始化质心
    '''
    centroids = init_centroids(df,k) # 方法一
    '''
    
    '''
    centroids = kmeansplus(df,k) # 方法二
    '''
    
    first_center = [np.random.randint(0,80), np.random.randint(0,80)]
    centroids = select_center(first_center, df, k) # 方法三
    
    # step 2: assign centroid for each source data
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    # 画数据点
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    # 画质心
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    # 迭代10次
    for i in range(10):
#        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main()



































