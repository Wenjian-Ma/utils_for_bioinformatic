import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.family']='SimHei'
# matplotlib.rcParams['font.sans-serif']='SimHei'

'''
[[0.802, 0.416, 0.632, 0.502, 0.398,0.819,0.488],
                [0.801, 0.411, 0.592, 0.485, 0.377,0.806,0.484],
                [0.755, 0.339, 0.583, 0.429, 0.304,0.764,0.393]]
'''

# [[0.802,0.801,0.755],
#  [0.416,0.411,0.339],
#  [0.632,0.592,0.583],
#  [0.502,0.485,0.429],
#  [0.398,0.377,0.304],
#  [0.819,0.806,0.764],
#  [0.488,0.484,0.393]]

dataset = pd.DataFrame(data=[[0.802,0.801,0.755],
 [0.416,0.411,0.339],
 [0.632,0.592,0.583],
 [0.502,0.485,0.429],
 [0.398,0.377,0.304],
 [0.819,0.806,0.764],
 [0.488,0.484,0.393]],
            index=['Acc','Pre','Recall ','F1 ','MCC','AUROC','AUPR'],
            columns=['Full', 'w/o SA-RIM','w/o EA-GCN'])
radar_labels=dataset.index
nAttr=7
data=dataset.values #数据值
data_labels=dataset.columns
# 设置角度
angles=np.linspace(0,2*np.pi,nAttr,
                   endpoint= False)
data=np.concatenate((data, [data[0]]))
angles=np.concatenate((angles, [angles[0]]))
# 设置画布
fig=plt.figure(facecolor="white",figsize=(10,6))
plt.subplot(111, polar=True)
# 绘图
plt.plot(angles,data,'o-',
         linewidth=1.5, alpha= 0.2)
# 填充颜色
plt.fill(angles,data, alpha=0.25)
plt.thetagrids(angles[:-1]*180/np.pi,
               radar_labels,1.2)
# plt.figtext(0.52, 0.95,'大学生通识能力分析',
#             ha='center', size=20)
# 设置图例
legend=plt.legend(data_labels,
                  loc=(1.1, 0.05),
                  labelspacing=0.1)
plt.setp(legend.get_texts(),
         fontsize='large')
plt.grid(linestyle='--')
plt.savefig('/media/ST-18T/Ma/LGS-PPIS/curve/Ablation.jpg', dpi=300)
# plt.show()
