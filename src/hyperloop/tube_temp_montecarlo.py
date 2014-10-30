import os
import time

import numpy as np
from math import log, pi, sqrt, e

from openmdao.main.api import Assembly, Component
from openmdao.lib.datatypes.api import Float, Bool
from openmdao.lib.drivers.api import CaseIteratorDriver, BroydenSolver
from openmdao.lib.casehandlers.api import BSONCaseRecorder

class HyperloopMonteCarlo(Assembly): 

    def configure(self): 

        driver = self.add('driver', CaseIteratorDriver())
        self.add('hyperloop', MiniHyperloop())

        driver.add_parameter('hyperloop.temp_outside_ambient')
        driver.add_parameter('hyperloop.solar_insolation')
        driver.add_parameter('hyperloop.surface_reflectance')
        driver.add_parameter('hyperloop.num_pods')
        driver.add_parameter('hyperloop.emissivity_tube')
        driver.add_parameter('hyperloop.Nu_multiplier')
        driver.add_parameter('hyperloop.compressor_adiabatic_eff')

        driver.add_response('hyperloop.temp_boundary')
        driver.add_response('hyperloop.radius_tube_outer')

        N_SAMPLES = 10
        driver.case_inputs.hyperloop.temp_outside_ambient = np.random.normal(305,10,N_SAMPLES)        
        driver.case_inputs.hyperloop.solar_insolation = np.random.triangular(200,1000,1000,N_SAMPLES); #left, mode, right, samples
        driver.case_inputs.hyperloop.surface_reflectance = np.random.triangular(0.5,0.7,1,N_SAMPLES);
        driver.case_inputs.hyperloop.num_pods = np.random.normal(34,2,N_SAMPLES);
        driver.case_inputs.hyperloop.emissivity_tube = np.random.triangular(0.4,0.4,0.9,N_SAMPLES);
        driver.case_inputs.hyperloop.Nu_multiplier = np.random.triangular(0.9,1,3,N_SAMPLES);
        driver.case_inputs.hyperloop.compressor_adiabatic_eff = np.random.triangular(0.6,0.69,0.8,N_SAMPLES);

        # driver.case_inputs.hyperloop.temp_outside_ambient = np.random.normal(305,10,N_SAMPLES)        
        # driver.case_inputs.hyperloop.solar_insolation = np.random.triangular(500,1000,1000,N_SAMPLES); #left, mode, right, samples
        # driver.case_inputs.hyperloop.surface_reflectance = np.random.triangular(0.7,0.85,1,N_SAMPLES);
        # driver.case_inputs.hyperloop.num_pods = np.random.normal(34,2,N_SAMPLES);
        # driver.case_inputs.hyperloop.emissivity_tube = np.random.triangular(0.4,0.4,0.6,N_SAMPLES);
        # driver.case_inputs.hyperloop.Nu_multiplier = np.random.triangular(0.9,1,3,N_SAMPLES);
        # driver.case_inputs.hyperloop.compressor_adiabatic_eff = np.random.triangular(0.6,0.69,0.8,N_SAMPLES);

        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.recorders = [BSONCaseRecorder('therm_mc_%s.bson'%timestamp)]

class MiniHyperloop(Assembly): 
    """ Abriged Hyperloop Model """ 
    def configure(self): 
        #Add Components
        self.add('tubeTemp', TubeWallTemp2())
        driver = self.add('driver',BroydenSolver())
        driver.add_parameter('tubeTemp.temp_boundary',low=0.,high=10000.)
        driver.add_constraint('tubeTemp.ss_temp_residual=0')
        driver.workflow.add(['tubeTemp'])

        #Boundary Input Connections
        #Hyperloop -> Compressor
        self.create_passthrough('comp.compressor_adiabatic_eff')

        #Hyperloop -> TubeWallTemp
        self.create_passthrough('tube_wall_temp.temp_outside_ambient')
        self.create_passthrough('tube_wall_temp.solar_insolation')
        self.create_passthrough('tube_wall_temp.surface_reflectance')
        self.create_passthrough('tube_wall_temp.num_pods')
        self.create_passthrough('tube_wall_temp.emissivity_tube')
        self.create_passthrough('tube_wall_temp.Nu_multiplier')

        #Inter-component Connections
        #Compress -> TubeWallTemp
        self.connect('comp.nozzle_Fl_O', 'tube_wall_temp.nozzle_air')
        self.connect('comp.bearing_Fl_O', 'tube_wall_temp.bearing_air')

class TubeWallTemp2(Component):
    """ [Tweaked from original to include simple comp calcs] Calculates Q released/absorbed by the hyperloop tube """

    #--New Comp Inputs--
    pod_MN = Float(0.91, iotype='in', desc='Capsule Mach number') 
    Wdot = Float(0.49, units='kg/s', iotype='in', desc='Airflow')
    tube_P = Float(99., units='Pa', iotype='in', desc='Tube ambient pressure')
    compPR = Float(12., iotype='in',desc='Compressor Pressure ratio')
    compressor_adiabatic_eff = Float(.8, iotype="in", desc="adiabatic efficiency for the compressors")

    inlet_Tt = Float(367, units='K', iotype='out', desc='Inlet total temperature')
    inlet_Pt = Float(169., units='Pa', iotype='out', desc='Compressor inlet total pressure')
    exit_Tt = Float(948, units='K', iotype='out', desc='Exit total temperature')
    exit_Pt = Float(2099., units='Pa', iotype='out', desc='Compressor exit total pressure')
    cp_air = Float(1148.9, units='J/(kg*K)', iotype='out', desc='Specific heat of air, compressor exit')
    pod_heat = Float(356149., units='W', iotype='out', desc='Heating due to a single capsule')
    #--Inputs--
    #Hyperloop Parameters/Design Variables
    radius_outer_tube = Float(1.115, units = 'm', iotype='in', desc='tube outer diameter') #7.3ft
    length_tube = Float(482803, units = 'm', iotype='in', desc='Length of entire Hyperloop') #300 miles, 1584000ft
    num_pods = Float(34, iotype='in', desc='Number of Pods in the Tube at a given time') #
    temp_boundary = Float(322.0, units = 'K', iotype='in', desc='Average Temperature of the tube wall') #
    temp_outside_ambient = Float(305.6, units = 'K', iotype='in', desc='Average Temperature of the outside air') #
    #nozzle_air = FlowStationVar(iotype="in", desc="air exiting the pod nozzle", copy=None)
    #bearing_air = FlowStationVar(iotype="in", desc="air exiting the air bearings", copy=None)

    #constants
    solar_insolation = Float(1000., iotype="in", units = 'W/m**2', desc='solar irradiation at sea level on a clear day') #
    nn_incidence_factor = Float(0.7, iotype="in", desc='Non-normal incidence factor') #
    surface_reflectance = Float(0.5, iotype="in", desc='Solar Reflectance Index') #
    q_per_area_solar = Float(350., units = 'W/m**2', desc='Solar Heat Rate Absorbed per Area') #
    q_total_solar = Float(375989751., iotype="in", units = 'W', desc='Solar Heat Absorbed by Tube') #
    emissivity_tube = Float(0.5, iotype="in", units = 'W', desc='Emmissivity of the Tube') #
    sb_constant = Float(0.00000005670373, iotype="in", units = 'W/((m**2)*(K**4))', desc='Stefan-Boltzmann Constant') #
    Nu_multiplier = Float(1, iotype="in", desc="fudge factor on nusslet number to account for small breeze on tube")

    #--Outputs--
    area_rad = Float(337486.1, units = 'm**2', iotype='out', desc='Tube Radiating Area') #    
    #Required for Natural Convection Calcs
    GrDelTL3 = Float(1946216.7, units = '1/((ft**3)*F)', iotype='out', desc='Heat Radiated to the outside') #
    Pr = Float(0.707, iotype='out', desc='Prandtl') #
    Gr = Float(12730351223., iotype='out', desc='Grashof #') #
    Ra = Float(8996312085., iotype='out', desc='Rayleigh #') #
    Nu = Float(232.4543713, iotype='out', desc='Nusselt #') #
    k = Float(0.02655, units = 'W/(m*K)', iotype='out', desc='Thermal conductivity') #
    h = Float(0.845464094, units = 'W/((m**2)*K)', iotype='out', desc='Heat Radiated to the outside') #
    area_convection = Float(3374876.115, units = 'W', iotype='out', desc='Convection Area') #
    #Natural Convection
    q_per_area_nat_conv = Float(7.9, units = 'W/(m**2)', iotype='out', desc='Heat Radiated per Area to the outside') #
    total_q_nat_conv = Float(286900419., units = 'W', iotype='out', desc='Total Heat Radiated to the outside via Natural Convection') #
    #Exhausted from Pods
    heat_rate_pod = Float(519763, units = 'W', iotype='out', desc='Heating Due to a Single Pods') #
    total_heat_rate_pods = Float(17671942., units = 'W', iotype='out', desc='Heating Due to a All Pods') #
    #Radiated Out
    q_rad_per_area = Float(31.6, units = 'W/(m**2)', iotype='out', desc='Heat Radiated to the outside') #
    q_rad_tot = Float(106761066.5, units = 'W', iotype='out', desc='Heat Radiated to the outside') #
    #Radiated In
    viewing_angle = Float(1074256, units = 'm**2', iotype='out', desc='Effective Area hit by Sun') #
    #Total Heating
    q_total_out = Float(286900419., units = 'W', iotype='out', desc='Total Heat Released via Radiation and Natural Convection') #
    q_total_in = Float(286900419., units = 'W', iotype='out', desc='Total Heat Absorbed/Added via Pods and Solar Absorption') #
    #Residual (for solver)
    ss_temp_residual = Float(units = 'K', iotype='out', desc='Residual of T_released - T_absorbed')
  
    def execute(self):
        """Calculate Various Paramters"""

        #New Simple Compressor Calcs
        self.inlet_Tt = self.tubeTemp*(1+0.2*self.pod_MN**2)
        self.inlet_Pt = self.tube_P*(1+0.2*self.pod_MN**2)**3.5

        self.exit_Tt = self.inlet_Tt*(1 + (1/self.compressor_adiabatic_eff)*(self.compPR**(1/3.5)-1) )
        self.exit_Pt = self.inlet_Pt * self.compPR

        if(self.exit_Tt<400):
            self.cp_air = 990.8*self.exit_Tt**(0.00316)
        else:
            self.cp_air = 299.4*self.exit_Tt**(0.1962)
        
        self.heat_rate_pod = self.W*self.cp_air*(self.exit_Tt-self.temp_boundary)
        #----
        self.diameter_outer_tube = 2*self.radius_outer_tube
        #bearing_q = cu(self.bearing_air.W,'lbm/s','kg/s') * cu(self.bearing_air.Cp,'Btu/(lbm*degR)','J/(kg*K)') * (cu(self.bearing_air.Tt,'degR','degK') - self.temp_boundary)
        #nozzle_q = cu(self.nozzle_air.W,'lbm/s','kg/s') * cu(self.nozzle_air.Cp,'Btu/(lbm*degR)','J/(kg*K)') * (cu(self.nozzle_air.Tt,'degR','degK') - self.temp_boundary)
        #Q = mdot * cp * deltaT 
        #self.heat_rate_pod = nozzle_q +bearing_q 
        #Total Q = Q * (number of pods)
        self.total_heat_rate_pods = self.heat_rate_pod*self.num_pods

        #Determine thermal resistance of outside via Natural Convection or forced convection
        if(self.temp_outside_ambient < 400):
            self.GrDelTL3 = 41780000000000000000*((self.temp_outside_ambient)**(-4.639)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            self.GrDelTL3 = 4985000000000000000*((self.temp_outside_ambient)**(-4.284)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        
        #Prandtl Number
        #Pr = viscous diffusion rate/ thermal diffusion rate = Cp * dyanamic viscosity / thermal conductivity
        #Pr << 1 means thermal diffusivity dominates
        #Pr >> 1 means momentum diffusivity dominates
        if (self.temp_outside_ambient < 400):
            self.Pr = 1.23*(self.temp_outside_ambient**(-0.09685)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            self.Pr = 0.59*(self.temp_outside_ambient**(0.0239))
        #Grashof Number
        #Relationship between buoyancy and viscosity
        #Laminar = Gr < 10^8
        #Turbulent = Gr > 10^9
        self.Gr = self.GrDelTL3*abs(self.temp_boundary-self.temp_outside_ambient)*(self.diameter_outer_tube**3) #JSG: Added abs incase subtraction goes negative
        #Rayleigh Number 
        #Buoyancy driven flow (natural convection)
        self.Ra = self.Pr * self.Gr
        #Nusselt Number
        #Nu = convecive heat transfer / conductive heat transfer
        if (self.Ra<=10**12): #valid in specific flow regime
            self.Nu = self.Nu_multiplier*((0.6 + 0.387*self.Ra**(1./6.)/(1 + (0.559/self.Pr)**(9./16.))**(8./27.))**2) #3rd Ed. of Introduction to Heat Transfer by Incropera and DeWitt, equations (9.33) and (9.34) on page 465
        if(self.temp_outside_ambient < 400):
            self.k = 0.0001423*(self.temp_outside_ambient**(0.9138)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            self.k = 0.0002494*(self.temp_outside_ambient**(0.8152))
        #h = k*Nu/Characteristic Length
        self.h = (self.k * self.Nu)/ self.diameter_outer_tube
        #Convection Area = Surface Area
        self.area_convection = pi * self.length_tube * self.diameter_outer_tube 
        #Determine heat radiated per square meter (Q)
        self.q_per_area_nat_conv = self.h*(self.temp_boundary-self.temp_outside_ambient)
        #Determine total heat radiated over entire tube (Qtotal)
        self.total_q_nat_conv = self.q_per_area_nat_conv * self.area_convection
        #Determine heat incoming via Sun radiation (Incidence Flux)
        #Sun hits an effective rectangular cross section
        self.area_viewing = self.length_tube* self.diameter_outer_tube
        self.q_per_area_solar = (1-self.surface_reflectance)* self.nn_incidence_factor * self.solar_insolation
        self.q_total_solar = self.q_per_area_solar * self.area_viewing
        #Determine heat released via radiation
        #Radiative area = surface area
        self.area_rad = self.area_convection
        #P/A = SB*emmisitivity*(T^4 - To^4)
        self.q_rad_per_area = self.sb_constant*self.emissivity_tube*((self.temp_boundary**4) - (self.temp_outside_ambient**4))
        #P = A * (P/A)
        self.q_rad_tot = self.area_rad * self.q_rad_per_area
        #------------
        #Sum Up
        self.q_total_out = self.q_rad_tot + self.total_q_nat_conv
        self.q_total_in = self.q_total_solar + self.total_heat_rate_pods
        
        self.ss_temp_residual = (self.q_total_out - self.q_total_in)/1e6

if __name__ == "__main__": 

    hl_mc = HyperloopMonteCarlo()

    #parameters
    hl_mc.hyperloop.Mach_bypass = .95
    hl_mc.hyperloop.Mach_pod_max = .8
    hl_mc.hyperloop.Mach_c1_in = .65
    hl_mc.hyperloop.c1_PR_des = 13

    #initial guesses
    hl_mc.hyperloop.compress.W_in = .38
    hl_mc.hyperloop.flow_limit.radius_tube = hl_mc.hyperloop.pod.radius_tube_inner = 243
    hl_mc.hyperloop.compress.Ts_tube = hl_mc.hyperloop.flow_limit.Ts_tube = hl_mc.hyperloop.tube_wall_temp.tubeWallTemp = 322.28
    hl_mc.hyperloop.compress.c2_PR_des = 8.72

    #initial run to converge things
    hl_mc.run()


