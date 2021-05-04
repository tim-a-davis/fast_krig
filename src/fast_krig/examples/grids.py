from fast_krig.examples.logs import generate_fake_log
from fast_krig.grid import Grid


fake_logs = [
    generate_fake_log(9000, 10000, 1, 0.2, 2000, log=True, name="RESISTIVITY")
    for i in range(50)
]

self = Grid(fake_logs, stream="RESISTIVITY", z_range=(9900, 9905))



"""


fig, ax = plt.subplots()
ax.plot(fake_logs[0].RESISTIVITY, fake_logs[0].index)
plt.gca().invert_yaxis()
ax.set_xscale('log')
plt.show()



"""