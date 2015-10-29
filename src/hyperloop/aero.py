from openmdao.core import Component, Problem, Group

#put inside pod
class Aero(Component): 
    """Place holder for real aerodynamic calculations of the capsule""" 
    def __init__(self):
        super(Aero, self).__init__()
        #Inputs
        self.add_param('coef_drag', 1.0, desc="capsule drag coefficient")
        self.add_param('area_capsule', 18000, units="cm**2", desc="capsule frontal area")
        self.add_param('velocity_capsule', 600, units="m/s", desc="capsule velocity")
        self.add_param('rho', 0.01, units="kg/m**3", desc="tube air density")
        self.add_param('gross_thrust', 10, units="N", desc="nozzle gross thrust") 
        #Outputs
        self.add_output('net_force', shape=1, units="N", desc="Net force with drag considerations")
        self.add_output('drag', shape=1, units="N", desc="Drag Force")

    def solve_nonlinear(self, params, unknowns, resids): 

        #Drag = 0.5*Cd*rho*Veloc*Area
        unknowns['drag'] = 0.5*params['coef_drag']*params['rho']*params['velocity_capsule']**2*params['area_capsule'] 
        unknowns['net_force'] = params['gross_thrust'] - unknowns['drag']

if __name__ == "__main__": 

    prob = Problem(root=Group())
    prob.root.add('aero', Aero())
    prob.setup()
    prob.run()

    print prob.root.unknowns['aero.drag']