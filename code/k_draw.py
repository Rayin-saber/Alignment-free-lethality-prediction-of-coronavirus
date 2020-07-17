import matplotlib.pyplot as plt
import numpy as np


# validation accuracy of 7 models
k_resnet = [0.970, 0.944, 0.803, 0.911, 0.862, 0.934, 0.974]
k_alexnet = [0.979, 0.991, 0.995, 0.993, 0.988, 0.988, 0.981]
k_vgg = [0.970, 0.981, 0.998, 1, 1, 0.993, 0.984]
k_knn = [0.993, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993]
k_rf = [0.991, 0.995, 1, 0.998, 1, 0.995, 0.998]
k_nn = [0.993, 0.993, 0.995, 1, 1, 1, 1]
k_lr = [0.984, 0.995, 0.995, 0.993, 0.993, 0.995, 0.995]

x_axis = range(1, 8)
fig = plt.figure()


plt.subplot(1, 2, 1)
# plt.title('Deep Learning Models')
plt.plot(x_axis, k_resnet, color='green', marker='o', ms=3, label='ResNet-34')
plt.plot(x_axis, k_alexnet, color='red', marker='o', ms=3, label='AlexNet')
plt.plot(x_axis, k_vgg,  color='skyblue', marker='o', ms=3, label='VGG-19')
plt.legend(loc='lower right')
plt.ylabel('Validation Accuracy')
plt.xlabel('K mers')
a = ['%.2f'%oi for oi in np.linspace(0.8, 1, 6)]
b =[eval(oo) for oo in a]
c = ['%d'%oi for oi in np.linspace(1, 7, 7)]
d =[eval(oo) for oo in c]
plt.yticks(b,a)
plt.xticks(d,c)
plt.ylim((0.8, 1.005))

plt.subplot(1, 2, 2)
# plt.title('Traditional Learning Models')
plt.plot(x_axis, k_lr, color='blue', marker='o', ms=3, label='LR')
plt.plot(x_axis, k_rf, color='pink', marker='o', ms=3, label='RF')
plt.plot(x_axis, k_nn,  color='yellow', marker='o', ms=3, label='NN')
plt.plot(x_axis, k_knn, color='lightgreen', marker='o', ms=3, label='KNN')


plt.xlabel('K mers')
plt.legend(loc='lower right')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.yaxis.grid()
ax2.yaxis.grid()
a = ['%.2f'%oi for oi in np.linspace(0.95, 1, 11)]
b =[eval(oo) for oo in a]
plt.yticks(b,a)
plt.xticks(d,c)
plt.ylim((0.95, 1.00125))
#plt.show()
plt.savefig("k.eps", dpi=500, format="eps")
