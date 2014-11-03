import numpy as np 
from matplotlib import pylab as plt
from openmdao.lib.casehandlers.api import CaseDataset
from matplotlib import mlab as mlab

cds = CaseDataset('../output/therm_mc_20141030140747.bson', 'bson')
data = cds.data.driver('driver').by_variable().fetch()
#temp
temp_boundary_k = data['hyperloop.temp_boundary']
temp_boundary = [((x-273.15)*1.8 + 32) for x in temp_boundary_k]
#histogram
n, bins, patches = plt.hist(temp_boundary, 50, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
#stats
mean = np.average(temp_boundary)
std = np.std(temp_boundary)
percentile = np.percentile(temp_boundary,85)
print "mean: ", mean, " std: ", std, " 85percentile: ", percentile
x = np.linspace(50,170,150)
plt.plot(x,mlab.normpdf(x,mean,std), color='blue')
plt.show()



