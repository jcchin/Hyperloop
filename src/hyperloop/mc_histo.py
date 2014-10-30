import numpy as np 

from matplotlib import pylab as plt

from openmdao.lib.casehandlers.api import CaseDataset

cds = CaseDataset('therm_mc_20141030115510.bson', 'bson')
data = cds.data.driver('driver').by_variable().fetch()

# temp_outside_ambient = np.loadtxt('temp_outside_ambient.txt')

# temp_boundary = np.loadtxt('temp_boundary.txt')
# radius_tube_outer = np.loadtxt('radius_tube_outer.txt')

#temp_outside_ambient = data['hyperloop.temp_outside_ambient']

#print "cases", temp_outside_ambient

temp_boundary_k = data['hyperloop.temp_boundary']

temp_boundary = [((x-273.15)*1.8 + 32) for x in temp_boundary_k]
#radius_tube_outer = data['hyperloop.radius_tube_outer']


n, bins, patches = plt.hist(temp_boundary, 50, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)


mode = np.average(temp_boundary)
std = np.std(temp_boundary)

percentile = np.percentile(temp_boundary,87)
print "mode: ", mode, " std: ", std, " 87percentile: ", percentile

plt.show()



