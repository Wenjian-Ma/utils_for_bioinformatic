import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
config = {
            #"font.family": 'serif',
            "font.size": 23,
           # "mathtext.fontset": 'stix',
           #"font.sans-serif": ['SimHei'],#宋体
            "font.family":'Times New Roman',
           # 'axes.unicode_minus': False # 处理负号
         }
rcParams.update(config)
#plt.rcParams['font.sans-serif'] = ['KaiTi']

label_Test_60 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/label_Test_60.npy')
score_Test_60 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/score_Test_60.npy')

label_Test_315 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/label_Test_315.npy')
score_Test_315 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/score_Test_315.npy')

label_Btest_31 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/label_Btest_31.npy')
score_Btest_31 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/score_Btest_31.npy')

label_UBtest_31 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/label_UBtest_31.npy')
score_UBtest_31 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/score_UBtest_31.npy')

# label01 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/label01.npy')
# score01 = np.load('/media/ST-18T/Ma/LGS-PPIS/curve/score01.npy')



# fpr_Test_60, tpr_Test_60, threshold_Test_60 = roc_curve(label_Test_60, score_Test_60)
# auc_Test_60 = auc(fpr_Test_60, tpr_Test_60)  # 计算AUC值
precision_Test_60, recall_Test_60, thresholds_Test_60 = precision_recall_curve(label_Test_60, score_Test_60)
auc_Test_60 = auc(recall_Test_60, precision_Test_60)  # 计算AUC值

# fpr_Test_315, tpr_Test_315, threshold_Test_315 = roc_curve(label_Test_315, score_Test_315)
#auc_Test_315 = auc(fpr_Test_315, tpr_Test_315)  # 计算AUC值
precision_Test_315, recall_Test_315, thresholds_Test_315 = precision_recall_curve(label_Test_315, score_Test_315)
auc_Test_315 = auc(recall_Test_315, precision_Test_315)  # 计算AUC值

# fpr_Btest_31, tpr_Btest_31, threshold_Btest_31 = roc_curve(label_Btest_31, score_Btest_31)
#auc_Btest_31 = auc(fpr_Btest_31, tpr_Btest_31)  # 计算AUC值
precision_Btest_31, recall_Btest_31, thresholds_Btest_31 = precision_recall_curve(label_Btest_31, score_Btest_31)
auc_Btest_31 = auc(recall_Btest_31, precision_Btest_31)  # 计算AUC值

# fpr_Btest_31, tpr_Btest_31, threshold_Btest_31 = roc_curve(label_Btest_31, score_Btest_31)
# auc_Btest_31 = auc(fpr_Btest_31, tpr_Btest_31)  # 计算AUC值

# fpr_UBtest_31, tpr_UBtest_31, threshold_UBtest_31 = roc_curve(label_UBtest_31, score_UBtest_31)
#auc_UBtest_31 = auc(fpr_UBtest_31, tpr_UBtest_31)  # 计算AUC值
precision_UBtest_31, recall_UBtest_31, thresholds_UBtest_31 = precision_recall_curve(label_UBtest_31, score_UBtest_31)
auc_UBtest_31 = auc(recall_UBtest_31, precision_UBtest_31)  # 计算AUC值

line_width = 1.5 # 曲线的宽度
plt.figure(figsize=(10, 7))  # 图的大小

plt.plot(recall_Test_60, precision_Test_60, lw=line_width, label='Test_60 (AUPR = %0.3f)' % auc_Test_60, color='red')

plt.plot(recall_Test_315, precision_Test_315, lw=line_width, label='Test_315 (AUPR = %0.3f)' % auc_Test_315, color='blue')

plt.plot(recall_Btest_31, precision_Btest_31, lw=line_width, label='Btest_31 (AUPR = %0.3f)' % auc_Btest_31, color='green')

plt.plot(recall_UBtest_31, precision_UBtest_31, lw=line_width, label='UBtest_31 (AUPR = %0.3f)' % auc_UBtest_31, color='orange')

# plt.plot(fpr01, tpr01, lw=line_width, label='Multi-species-1%% (AUROC = %0.4f)' % auc01, color='grey')


plt.xlim([0.0, 1.0])  # 限定x轴的范围
plt.ylim([0.0, 1.0])  # 限定y轴的范围

# y_ticks = [0.0,0.980,0.985,0.990,0.995,1.000]

# plt.xticks(range(0, 10, 1)) # 修改x轴的刻度
# plt.yticks(range(0, 10, 1)) # 修改y轴的刻度

plt.xlabel('Recall')  # x坐标轴标题
plt.ylabel('Precision')  # y坐标轴标题
# plt.xlabel('Recall')
# plt.ylabel('Precision')

# plt.title('Receiver Operating Characteristic')  # 图标题

plt.grid(linestyle='--')  # 在图中添加网格

plt.legend(loc="upper right")  # 显示图例并指定图例位置


# 5.中文处理问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


# 6.展示图片和保存
plt.savefig('/media/ST-18T/Ma/LGS-PPIS/curve/PR.jpg', dpi=300)
