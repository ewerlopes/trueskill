import pickle
import numpy as np
from matplotlib import pyplot as plt


data = pickle.load(open("plk/game1.pkl", "rb"))

white = data["white"]
black = data["black"]

w_cpl = [0 if float(white["cpl"][x]) < 0
         else float(white["cpl"][x]) for x in range(len(white["cpl"]))]

b_cpl = [0 if float(black["cpl"][x]) < 0
         else float(black["cpl"][x]) for x in range(len(black["cpl"]))]

diff = []
for i in range(len(white["cpl"])):
    diff.append(w_cpl[i]-b_cpl[i])

w_emt = [float(white["emt"][x]) for x in range(len(white["emt"]))]
b_emt = [float(black["emt"][x]) for x in range(len(black["emt"]))]

diff_emt = []
for i in range(len(w_emt)):
    try:
        diff_emt.append(w_emt[i]-b_emt[i])
    except Exception as e:
        print e

print("Length: {}".format(len(white["cpl"])))
plt.figure(figsize=(8, 8))
plt.subplot(231)
plt.scatter(range(len(white["cpl"])),
            w_cpl,
            c='b', marker="x", label='white')
plt.scatter(range(len(black["cpl"])),
            b_cpl,
            c='r', marker="o", label='black')
plt.legend(loc='upper left', fontsize=8)
plt.ylabel("centipawn loss")
plt.xlabel("moves")

plt.subplot(232)
plt.scatter(range(len(white["emt"])),
            w_emt,
            c='b', marker="x", label='white')
plt.scatter(range(len(black["emt"])),
            b_emt,
            c='r', marker="o", label='black')

plt.legend(loc='upper left', fontsize=8)
plt.ylabel("emt")
plt.xlabel("moves")
#plt.yscale('log')

plt.subplot(233)
#plt.hist(w_cpl, bins=50, histtype='stepfilled', normed=True, color='b', label='white')
#plt.hist(b_cpl, bins=50, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='black')
plt.hist(diff, bins=50, histtype='stepfilled', normed=True, color='g', alpha=0.5, label='diff')
plt.xlabel("centipawn loss")
plt.ylabel("Probability")
#plt.xlim([0,200])
plt.legend(loc='best', fontsize=8)

plt.subplot(234)
#plt.hist(w_emt, bins=50, histtype='stepfilled', normed=True, color='b', label='white')
#plt.hist(b_emt, bins=50, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='black')
plt.hist(diff_emt, bins=50, histtype='stepfilled', normed=True, color='g', alpha=0.5, label='diff')
plt.xlabel("emt")
plt.ylabel("Probability")
plt.legend(loc='best', fontsize=8)


plt.subplot(235)
s = np.random.poisson(w_cpl[3], 10000)
count, bins, ignored = plt.hist(s, 14, normed=True)

plt.subplot(236)
s = np.random.poisson(b_cpl[3], 10000)
count, bins, ignored = plt.hist(s, 14, normed=True)

plt.show()