import matplotlib.pyplot as plt
import numpy as np


dbase = np.recfromcsv('tally.csv', encoding='UTF-8')

dy = dbase['dy']
dm = dbase['dm']
dr = dbase['dr']
do = dbase['do']
d = dbase['d']

cfr = 15.9


dc = dy+dm+dr+do

cfry = dy[-1]/dc[-1] * cfr
print(f'cfr 0-45: {cfry} %')

days = np.array([int(el) for el in dbase['dag']])


plt.fill_between( days, 0*dy, dy, color='C0', alpha=0.3, label='0-45' )
plt.fill_between( days, dy, dy+dm, color='C1', alpha=0.3, label='45-65')
plt.fill_between( days, dy+dm, dy+dm+dr, color='C2', alpha=0.3, label='65-75')
plt.fill_between( days, dy+dm+dr, dy+dm+dr+do, color='C3', alpha=0.3, label='75+')
plt.fill_between( days, dy+dm+dr+do,d, color=(0.7,0.7,0.7), alpha=0.3, label='N/A')
plt.xlabel('Dag (na 2020-03-15)', fontsize=16)
plt.ylabel('# Doden', fontsize=16)
plt.legend(frameon=False, fontsize=16, loc=2)
plt.tight_layout()
plt.savefig('mortality_age.png', dpi=300)
