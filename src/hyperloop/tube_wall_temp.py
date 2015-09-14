"""
    tubeModel.py -
        Determines the steady state temperature of the hyperloop tube.
        Calculates Q released/absorbed by hyperloop tube due to:
        Internal Convection, Tube Conduction, Ambient Natural Convection, Solar Flux In, Radiation Out

    -original calculations from Jeff Berton, ported and extended by Jeff Chin

    Compatible with OpenMDAO v0.8.1
"""
from math import log, pi, sqrt, e

from openmdao.core import Problem, Group, Component
from openmdao.solvers import Newton
from openmdao.units import convert_units as cu

from pycycle import species_data
from pycycle.constants import AIR_FUEL_MIX, AIR_MIX
from pycycle.flowstation import FlowIn, PassThrough

class TempBalance(Component):

    def __init__(self):
        super(TempBalance, self).__init__()
        self.add_param('ss_temp_residual', val=0.)
        self.add_state('temp_boundary', val=322.0)

    def solve_nonlinear(self, params, unknowns, resids):

        resids['temp_boundary'] = params['ss_temp_residual'] #drive ss_temp_residual to 0

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        if mode == "fwd":
            dresids['temp_boundary'] += dparams['ss_temp_residual']

        if mode == "fwd":
            dparams['ss_temp_residual'] += dresids['temp_boundary']


class TubeWallTemp(Component):
    """ Calculates Q released/absorbed by the hyperloop tube """
    def __init__(self, thermo_data=species_data.janaf, elements=AIR_MIX):
        super(TubeWallTemp, self).__init__()
        self.fd_options['force_fd'] = True

        self.thermo_data = thermo_data
        self.elements = elements
        gas_thermo = species_data.Thermo(self.thermo_data, init_reacts=self.elements)
        self.gas_prods = gas_thermo.products
        self.num_prod = len(self.gas_prods)
        flow_in = FlowIn('Fl_I', self.num_prod)
        #--Inputs--
        #Hyperloop Parameters/Design Variables
        self.add_param('radius_outer_tube', 1.115, units='m', desc='tube outer diameter') #7.3ft
        self.add_param('length_tube', 482803, units='m', desc='Length of entire Hyperloop') #300 miles, 1584000ft
        self.add_param('num_pods', 34, desc='Number of Pods in the Tube at a given time') #
        self.add_param('temp_boundary', 322.0, units='K', desc='Average Temperature of the tube wall') #
        self.add_param('temp_outside_ambient', 305.6, units='K', desc='Average Temperature of the outside air') #
        self.add('nozzle_air', flow_in, promotes=flow_in.flow_in_vars)
        self.add('bearing_air', flow_in, promotes=flow_in.flow_in_vars)
        #nozzle_air = FlowIn(iotype="in", desc="air exiting the pod nozzle")
        #bearing_air = FlowIn(iotype="in", desc="air exiting the air bearings")

        #constants
        self.add_param('solar_insolation', 1000., units='W/m**2', desc='solar irradiation at sea level on a clear day') #
        self.add_param('nn_incidence_factor', 0.7, desc='Non-normal incidence factor') #
        self.add_param('surface_reflectance', 0.5, desc='Solar Reflectance Index') #
        self.add_param('q_per_area_solar', 350., units='W/m**2', desc='Solar Heat Rate Absorbed per Area') #
        self.add_param('q_total_solar', 375989751., units='W', desc='Solar Heat Absorbed by Tube') #
        self.add_param('emissivity_tube', 0.5, units='W', desc='Emmissivity of the Tube') #
        self.add_param('sb_constant', 0.00000005670373, units='W/((m**2)*(K**4))', desc='Stefan-Boltzmann Constant') #
        self.add_param('Nu_multiplier', 1, desc="fudge factor on nusslet number to account for small breeze on tube")

        #--Outputs--
        self.add_output('area_rad', 337486.1, units='m**2', desc='Tube Radiating Area')  #   
        #Required for Natural Convection Calcs
        self.add_output('GrDelTL3', 1946216.7, units='1/((ft**3)*F)', desc='Heat Radiated to the outside') #
        self.add_output('Pr', 0.707, desc='Prandtl') #
        self.add_output('Gr', 12730351223., desc='Grashof #') #
        self.add_output('Ra', 8996312085., desc='Rayleigh #') #
        self.add_output('Nu', 232.4543713, desc='Nusselt #') #
        self.add_output('k', 0.02655, units='W/(m*K)', desc='Thermal conductivity') #
        self.add_output('h', 0.845464094, units='W/((m**2)*K)', desc='Heat Radiated to the outside') #
        self.add_output('area_convection', 3374876.115, units='W', desc='Convection Area') #
        #Natural Convection
        self.add_output('q_per_area_nat_conv', 7.9, units='W/(m**2)', desc='Heat Radiated per Area to the outside') #
        self.add_output('total_q_nat_conv', 286900419., units='W', desc='Total Heat Radiated to the outside via Natural Convection') #
        #Exhausted from Pods
        self.add_output('heat_rate_pod', 519763, units='W', desc='Heating Due to a Single Pods') #
        self.add_output('total_heat_rate_pods', 17671942., units='W', desc='Heating Due to a All Pods') #
        #Radiated Out
        self.add_output('q_rad_per_area', 31.6, units='W/(m**2)', desc='Heat Radiated to the outside') #
        self.add_output('q_rad_tot', 106761066.5, units='W', desc='Heat Radiated to the outside') #
        #Radiated In
        self.add_output('viewing_angle', 1074256, units='m**2', desc='Effective Area hit by Sun') #
        #Total Heating
        self.add_output('q_total_out', 286900419., units='W', desc='Total Heat Released via Radiation and Natural Convection') #
        self.add_output('q_total_in', 286900419., units='W', desc='Total Heat Absorbed/Added via Pods and Solar Absorption') #
        #Residual (for solver)
        self.add_output('ss_temp_residual', shape=1, units='K', desc='Residual of T_released - T_absorbed')
      
    def solve_nonlinear(self, params, unknowns, resids):
        """Calculate Various Paramters"""

        unknowns['diameter_outer_tube'] = 2*params['radius_outer_tube']
        
        unknowns['bearing_q'] = cu(params['bearing_air.W'],'lbm/s','kg/s') * cu(params['bearing_air.Cp'],'Btu/(lbm*degR)','J/(kg*K)') * (cu(params['bearing_air.Tt'],'degR','degK') - parparams['temp_boundary'])
        unknowns['nozzle_q'] = cu(params['nozzle_air.W'],'lbm/s','kg/s') * cu(params['nozzle_air.Cp'],'Btu/(lbm*degR)','J/(kg*K)') * (cu(params['nozzle_air.Tt'],'degR','degK') - parparams['temp_boundary'])
        #Q = mdot * cp * deltaT 
        unknowns['heat_rate_pod'] = unknowns['nozzle_q'] + unknowns['bearing_q']
        #Total Q = Q * (number of pods)
        unknowns['total_heat_rate_pods'] = unknowns['heat_rate_pod']*params['num_pods']

        #Determine thermal resistance of outside via Natural Convection or forced convection
        if(params['temp_outside_ambient'] < 400):
            unknowns['GrDelTL3'] = 41780000000000000000*((params['temp_outside_ambient'])**(-4.639)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            unknowns['GrDelTL3'] = 4985000000000000000*((params['temp_outside_ambient'])**(-4.284)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        
        #Prandtl Number
        #Pr = viscous diffusion rate/ thermal diffusion rate = Cp * dyanamic viscosity / thermal conductivity
        #Pr << 1 means thermal diffusivity dominates
        #Pr >> 1 means momentum diffusivity dominates
        if (params['temp_outside_ambient'] < 400):
            unknowns['Pr'] = 1.23*(params['temp_outside_ambient']**(-0.09685)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            unknowns['Pr'] = 0.59*(params['temp_outside_ambient']**(0.0239))
        #Grashof Number
        #Relationship between buoyancy and viscosity
        #Laminar = Gr < 10^8
        #Turbulent = Gr > 10^9
        unknowns['Gr'] = unknowns['GrDelTL3']*abs(params['temp_boundary']-params['temp_outside_ambient'])*(params['diameter_outer_tube']**3) #JSG: Added abs incase subtraction goes negative
        #Rayleigh Number 
        #Buoyancy driven flow (natural convection)
        unknowns['Ra'] = unknowns['Pr'] * unknowns['Gr']
        #Nusselt Number
        #Nu = convecive heat transfer / conductive heat transfer
        if (unknowns['Ra']<=10**12): #valid in specific flow regime
            unknowns['Nu'] = params['Nu_multiplier']*((0.6 + 0.387*unknowns['Ra']**(1./6.)/(1 + (0.559/unknowns['Pr'])**(9./16.))**(8./27.))**2) #3rd Ed. of Introduction to Heat Transfer by Incropera and DeWitt, equations (9.33) and (9.34) on page 465
        if(params['temp_outside_ambient'] < 400):
            unknowns['k'] = 0.0001423*(params['temp_outside_ambient']**(0.9138)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            unknowns['k'] = 0.0002494*(params['temp_outside_ambient']**(0.8152))
        #h = k*Nu/Characteristic Length
        unknowns['h'] = (unknowns['k'] * unknowns['Nu'])/ params['diameter_outer_tube']
        #Convection Area = Surface Area
        unknowns['area_convection'] = pi * params['length_tube'] * params['diameter_outer_tube'] 
        #Determine heat radiated per square meter (Q)
        unknowns['q_per_area_nat_conv'] = unknowns['h']*(params['temp_boundary']-params['temp_outside_ambient'])
        #Determine total heat radiated over entire tube (Qtotal)
        unknowns['total_q_nat_conv'] = unknowns['q_per_area_nat_conv'] * unknowns['area_convection']
        #Determine heat incoming via Sun radiation (Incidence Flux)
        #Sun hits an effective rectangular cross section
        unknowns['area_viewing'] = params['length_tube'] * params['diameter_outer_tube']
        unknowns['q_per_area_solar'] = (1-params['surface_reflectance'])* params['nn_incidence_factor'] * params['solar_insolation']
        unknowns['q_total_solar'] = unknowns['q_per_area_solar'] * unknowns['area_viewing']
        #Determine heat released via radiation
        #Radiative area = surface area
        unknowns['area_rad'] = unknowns['area_convection']
        #P/A = SB*emmisitivity*(T^4 - To^4)
        unknowns['q_rad_per_area'] = params['sb_constant']*params['emissivity_tube']*((params['temp_boundary']**4) - (params['temp_outside_ambient']**4))
        #P = A * (P/A)
        unknowns['q_rad_tot'] = unknowns['area_rad'] * unknowns['q_rad_per_area']
        #------------
        #Sum Up
        unknowns['q_total_out'] = unknowns['q_rad_tot'] + unknowns['total_q_nat_conv']
        unknowns['q_total_in'] = unknowns['q_total_solar'] + unknowns['total_heat_rate_pods']

        unknowns['ss_temp_residual'] = (unknowns['q_total_out'] - unknowns['q_total_in'])/1e6

#run stand-alone component
if __name__ == "__main__":

    root = Group()
    root.add('tm', TubeWallTemp())
    root.add('tmp_balance', TempBalance())
    prob = Problem(root)
    
    prob.root.connect('tm.ss_temp_residual','tmp_balance.ss_temp_residual')
    prob.root.connect('tmp_balance.temp_boundary','tm.temp_boundary')

    prob.root.nl_solver = Newton()
    prob.root.nl_solver.options['atol'] = 1e-3
    prob.root.nl_solver.options['iprint'] = 1
    prob.root.nl_solver.options['rtol'] = 1e-5
    prob.root.nl_solver.options['maxiter'] = 50


    params = (
        ('P', 17., {'units':'psi'}),
        ('T', 500.0, {'units':'degR'}),
        ('W', 1.0, {'units':'lbm/s'})
        )
    prob.root.add('des_vars', ParamComp(params))
    prob.root.connect('des_vars.P', 'tm.nozzle_air.P')
    prob.root.connect('des_vars.T', 'tm.nozzle_air.T')
    prob.root.connect('des_vars.W', 'tm.nozzle_air.W')
    prob.root.connect('des_vars.Ps_exhaust', 'turbojet.nozz.Ps_exhaust')

    prob['des_vars.T'] = 1710.0
    prob['des_vars.P'] = 0.304434211
    prob['des_vars.W'] = 1.08
    #set input values
    #prob.root.tm.nozzle_air.setTotalTP(1710, 0.304434211)
    #prob.root.tm['nozzle_air.W'] = 1.08
    prob.root.tm['bearing_air.W'] = 0.
    prob.root.tm['diameter_outer_tube'] = 2.22504#, units='m', iotype='in', desc='Tube out diameter') #7.3ft
    prob.root.tm['length_tube'] = 482803.#, units='m', iotype='in', desc='Length of entire Hyperloop') #300 miles, 1584000ft
    prob.root.tm['num_pods'] = 34.#, units='K', iotype='in', desc='Number of Pods in the Tube at a given time') #
    prob.root.tm['temp_boundary'] = 340#, units='K', iotype='in', desc='Average Temperature of the tube') #
    prob.root.tm['temp_outside_ambient'] = 305.6#, units='K', iotype='in', desc='Average Temperature of the outside air') #

    prob.run()

    print "temp_boundary: ", prob['root.tm.tmp_balance']

    # print "-----Completed Tube Heat Flux Model Calculations---"
    # print ""
    # print "CompressQ-{} SolarQ-{} RadQ-{} ConvecQ-{}".format(test.tm.total_heat_rate_pods, test.tm.q_total_solar, test.tm.q_rad_tot, test.tm.total_q_nat_conv )
    # print "Equilibrium Wall Temperature: {} K or {} F".format(tesparams['temp_boundary'], cu(tesparams['temp_boundary'],'degK','degF'))
    # print "Ambient Temperature:          {} K or {} F".format(test.tm.temp_outside_ambient, cu(test.tm.temp_outside_ambient,'degK','degF'))
    # print "Q Out = {} W  ==>  Q In = {} W ==> Error: {}%".format(test.tm.q_total_out,test.tm.q_total_in,((test.tm.q_total_out-test.tm.q_total_in)/test.tm.q_total_out)*100)