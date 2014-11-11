import numpy as np 
from matplotlib import pylab as plt
from openmdao.lib.casehandlers.api import CaseDataset
from matplotlib import mlab as mlab

class MC_Plot:
    def plot(self,file):
        cds = CaseDataset(file, 'bson')
        data = cds.data.driver('driver').by_variable().fetch()
        cds2 = CaseDataset('../output/therm_mc_20141110173851.bson', 'bson')
        data2 = cds2.data.driver('driver').by_variable().fetch()
        
        #temp
        temp_boundary_k = data['hyperloop.temp_boundary']
        temp_boundary_k.extend(data2['hyperloop.temp_boundary'])
        temp_boundary = [((x-273.15)*1.8 + 32) for x in temp_boundary_k]
        #histogram
        n, bins, patches = plt.hist(temp_boundary, 100, normed=1, histtype='stepfilled')
        plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)

        #stats
        mean = np.average(temp_boundary)
        std = np.std(temp_boundary)
        percentile = np.percentile(temp_boundary,99.5)
        print "mean: ", mean, " std: ", std, " 99.5percentile: ", percentile
        x = np.linspace(50,170,150)
        plt.plot(x,mlab.normpdf(x,mean,std), color='black', lw=2)
        plt.xlim([60,160])
        plt.ylabel('Probability', fontsize=18)
        plt.xlabel(u'Equilibrium Temperature, \N{DEGREE SIGN}F', fontsize=18)
        #plt.show()
        plt.tight_layout()
        plt.savefig('../output/histo.pdf', dpi=300)
if __name__ == "__main__": 

    p = MC_Plot()
    
    #p.plot('../output/therm_mc_20141110173851.bson')
    p.plot('../output/therm_mc_20141110194434.bson')