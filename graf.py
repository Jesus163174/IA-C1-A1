import matplotlib.pyplot as plt

fig = plt.figure(u'Gráfica de tiempos') # Figure
ax = fig.add_subplot(111) # Axes

nombres = ['SSD','SAD','CROSS']
datos = [70.98977279663086,63.46286988258362,55.13333773612976]
xx = range(len(datos))

ax.bar(xx, datos, width=0.8, align='center')
ax.set_xticks(xx)
ax.set_xticklabels(nombres)

plt.show()