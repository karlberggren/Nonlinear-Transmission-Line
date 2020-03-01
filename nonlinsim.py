"""
Nonlinsim

Has fully featured CLI with help.  General advice is timestep should
be much finer than position step (in units where c = 1).

A few examples:

Example 1: runs quickly
python3 nonlinsim.py --mur 1 --epsr 1 --length 8 --dx 1e-1 --dt 1e-3 --duration 8 --frames 10 --Rload 1 --Rin match --alpha .5 --beta .1 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -f test.txt -p

Example 2: for debugging
python3 nonlinsim.py --mur 1 --epsr 1 --length 1 --dx 1e-1 --dt 1e-3 --duration 1 --frames 10 --Rload 1 --Rin match --alpha 0 --beta 0 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -p

Example 3: including hotspot effects
python3 nonlinsim.py --mur 1 --epsr 1 --length 1 --dx 1e-2 --dt 1e-4 --duration 2 --frames 10 --Rload 1 --Rin 1 --alpha 0 --beta 0 --source step --amplitude 2e-6 --phase .5 --sigma .01 --offset 0 --ic 1e-3 --ihs 0.3e-3 --bias 0 --Rsheet 1 -p -f 2020-01-14-termination1.dat &

Adapative timestep --adaptt If energy loss/gain in system is more than
a fixed fraction of the total energy (say 1%) then throw away that
timestep, divide timestep by 2, and re-run.  If energy loss/gain in
system is less than a fixed fraction of the total energy (say .1%)
then keep that timestep, but set next timestep to be 2x current
timestep.

FIXME refactor out Vretrap and replace with a Rmin

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erf as erf
from scipy import constants

# These are physical constants... they're gonna be globals.  Sorry.
c = constants.speed_of_light
Zo = np.sqrt(constants.mu_0/constants.epsilon_0)

"""
Hotspot class


"""
class Hotspot:
    # some class attributes that will be accessed as constants
    psi = 38
    vo = 250 / c  # m/s
    f = 0.1

    # some class attributes that will be used by simulation, effectively as globals, but protected in class
    active = set()  # set of all active hotspots
    archive = set()  # set of all past hotspots
    
    def __init__(self, t_o, ndx, params,sim_params):
        self.index = ndx
        self.t_o = t_o
        self.hotspot_age = 0.0
        self.Rs = params['Rsheet']
        self.res = self.f * self.Rs
        self.width = params['width']
        self.V_retrap = sim_params['V_retrap']  # FIXME refactor
        """
        the history vector is the main way we will keep track of the state of the hotspot.
        each tuple will track current time, resistance, and power dissipated in previous timestep
        """
        self.history = np.array([(self.hotspot_age, self.res, 0)])
        Hotspot.active.add(self)
        return

    def step(self, i, dt):
        """
        update hotspot state by a time step of duration dt
        
        return True if hotspot has not collapsed
        return False if hotspot has collapsed

        """

        i = i / isw
        vhs = 2*vo*(psi*i**2 - 2) / np.sqrt(psi*i**2 -1)
        oldres = self.res
        self.res += vhs * dt * self.Rs / self.width
        self.hotspot_age += dt

        # power dissipated, use average of resistance between timesteps
        dp = i**2 * i_sw**2 * (oldres + self.res)/2.0
        
        if self.res <= self.V_retrap / (np.abs(i)*isw):  # hotspot collapsed?
            Hotspot.active.remove(self)
            Hotspot.archive.add(self)
        else:
            np.append(self.R_history, (self.hotspot_age, self.res, dp))
        return


    def delete_timestep(self):
        """
        delete last timestep (used in adaptive timestepping)
        """
        self.history = self.history[:-1]
        self.res = self.history[-1][2]
        self.hotspot_age = self.history[0]
        return
    
    """
    active_hotspot indices

    helper to generate list of indices of active hotspots
    """
    @staticmethod
    def active_hotspot_indices():
        indices = []
        for hotspot in Hotspot.active():
            indices.append(hotspot.index)
        return indices

    """
    get_hotspot

    helper to return hotspot at a given index, or return 'default' (False) if no
    hotspot exists at that index
    """
    @staticmethod
    def get_hotspot(ndx, default = False):
        for hotspot in Hotspot.active():
            if hotspot.index == ndx:
                return hotspot
        return default
            

"""
Create some helper functions, to make it easier to create some generic
inputs.  Of course, user can always make themselves a custom function
to pass as input.
"""
def gaussian(amplitude, t_offset, sigma, offset = 0):
    """
    generic gaussian function, for creating gaussian input pulse
    """
    def v_IN(t):
        return amplitude * np.exp(-(t-center)**2/(2*sigma**2)) + offset

    return v_IN

def sinusoid(amplitude, frequency, t_offset = 0, offset = 0):
    """
    generic sinusoidal function, for creating sinusoidal input
    """
    def v_IN(t):
        return amplitude * np.cos(2*np.pi*frequency*(t - t_offset)) + offset

    return v_IN

def step(amplitude, t_offset, width, offset = 0):
    """
    generic sigmoid function
    """
    def v_IN(t):
        return amplitude*(1/2+1/2*erf((t-t_offset)/width)) + offset

    return v_IN

def funcsum(func1, func2):
    """
    sum two of our other functions.  To use this, first create the functions
    using step, sinusoid, or gaussian, then pass those two functions to funcsum
    and it will return a function that is the sum.  No CLI yet.  Better for GUI.

    Sympy or similar probably has a mechanism for this, but don't know how.  This
    works fine.
    """
    def v_IN(t):
        return func1(t) + func2(t)

    return v_IN
    
"""
simulate

:: v_IN :: main input function of time
:: sim_params :: dict of simulation relevant parameters
:: params :: dict of physically relevant parameters

perform simulation over duration etc. as specified in parameter dictionaries

"""
def simulate(v_IN, sim_params, params):
    length = params["length"]
    duration = params["duration"]

    dx = sim_params["dx"]
    dt = sim_params["dt"]
    numframes = sim_params["numframes"]

    def mu(current):
        """
        current: current

        specify nonlinear dependence of mu on current.  
        """
        return params["mu_r"]*(1 + 2*params["alpha"]*current**2 +
                               4*params["beta"]*current**4)

    def mu_eff(current):
        """
        it turns out the i dL/dt effect on voltage can be folded in by
        using an "effective mu" (see notes of 2020-01-02).
        """
        return params["mu_r"] * (1 +
                                 3*params["alpha"]*current**2 +
                                 5*params["beta"]*current**4)

    t = 0.0  # tracks total simulation time

    """
    The frame_timer counts up with time until a frame duration is reached, then resets
    """

    frame_timer = 0.0
    frame_duration = duration/numframes
    frames = []  # array for storing frames as they are produced

    """
    Model this as an array of "numpoints" small inductors and capacitors.  
    """

    numpoints = int(length/dx)

    """
    i[0] refers to the current in the first inductor (from left to right), which is
    immediately preceded by the input resistor, and followed by a capacitor to 
    ground (v[0]).
    """
    v = [params["bias"]*params["ic"]*params["RL"] for _ in range(numpoints)]
    i = [params["bias"]*params["ic"] for _ in range(numpoints)]
    icnormd = [1 for _ in range(numpoints)]

    # suppress ic in one node, set by control parameters, as if photon hit there.
    
    if 1.0 >= params["hotspot_location"] >= 0 :
        icnormd[int(params["hotspot_location"] * numpoints)] = 0.8 

    """
    Now we'll iterate through the timesteps.

    First a helper function for that.
    """

    def timestep(i, v, left_boundary, right_boundary, dt):
        """
        move i,v vectors forward from time t to time t+dt, taking care of
        boundary conditions.

        update any hotspots

        return updated i, v
        """
        
        newi = []  # i(t+dt)
        newv = []  # v(t+dt)
        
        eps = params["eps_r"]

        # deal with left boundary
        if left_boundary['type'] == 'source':
            mus = mu_eff(i[0])
            Vin = left_boundary['source strength']
            RIN = left_boundary['source impedance']

            # The Vin-v[0] term can be very large... this could cause problems
            dv = Vin - v[0]

            """
                          v0    
            o--RIN--o--L--o--L-- . . .
            |         ->  |  ->
            Vin       i0  C  i1
            |             |
            g             g

            """
            dv, di = Vin - v[0] - i[0] * RIN, i[0] - i[1]
            
            newi.append(dt / mus / dx * dv + i[0])
            newv.append(dt / dx /eps * di + v[0])

        # deal with body of array
        
        for n in range(1,len(i)-1):  # iterate through points, avoiding endpoints
            mus = mu_eff(i[n])
            """
                 -> iR
         . . . o--R--o--R--o . . .
               |     |     |
         v(n-1)|     |vn   | 
         . . . o--L--o--L--o . . .
                 ->  |  ->
                 in  C  i(n+1)
                     |
                     g

            """
            dv = v[n-1] - v[n]  # calculate dv

            # calculate resistivity so that cutoff frequency will be as specified
            
            if sim_params['rho']:
                iR = -dv / (sim_params["rho"] * dx)  # current in resistor
            else:
                iR = 0
            di = i[n] + iR - i[n+1]  # current in capacitor
           
            # first do routine calculations, will overwrite if needed
            newi.append(dt/mus/dx*dv + i[n])
            newv.append(dt/dx/eps*di + v[n])
            
            # check if a hotspot needs to be created
            if abs(newi[n]) > params["ic"]*icnormd[n]:
                # switching current exceeded
                if n not in Hotspot.active_hotspot_indices():
                    # hotspot didn't exist at this location previously, create it
                    Hotspot(t, n, params, sim_params)
            else:
                # check if there's a hotspot here already
                hotspot = Hotspot.get_hotspot(n):
                if hotspot:
                    Rhs = hotspot.res
                    """

                 v(n-1)            v(n)
                 . . . o--L---Rhs--o
                         ->        |
                         in        C
                                   |
                                   g

                    """
                    dv = v[n-1] - (v[n] +  i[n] * Rhs)  # voltage across inductor
                    di = i[n] - i[n+1]  # current in capacitor

                    newi.append(dt/mus/dx*dv + i[n])
                    newv.append(dt/dx/eps*di + v[n])
                    hotspot.step(i[n],dt) == 0

        # deal with right boundary
        if right_boundary['type'] == 'load':
            Rload = right_boundary['load impedance']
            # tried on Jan 14 2020 to improve this termination condition
            # kludge Rload + .01 to avoid problem when term is shorted.

            """
                 -> iR
         . . . o--R--o    
               |     |     
          v[-2]|     |v[-1]    
         . . . o--L--o-----o
                 ->  |     |
               i[-1] C     Rl
                     |     |
                     g     g

            """
            # I've been having troubles when very large gradients are present.
            dv = v[-2] - v[-1]
            if sim_params['rho']:
                iR = - dv / (sim_params["rho"]*dx)
            else:
                iR = 0
            di = i[-1] + iR - v[-1]/(Rload + .01)
            newv.append(v[-1] + di * dt / eps / dx)
            newi.append(dv * dt / mus / dx + i[-1])
            
        return newi, newv
    

    def energy(i,v) :
        """
        calculate total energy stored in transmission line
        """
        Co = params["eps_r"] * dx
        nrg = 0
        Lo = params["mu_r"] * dx
        α = params["alpha"]
        β = params["beta"]
        for n,io in enumerate(i):
            # sum all inductors
            nrg += Lo * (0.5*io**2 +
                         2/3*α*io**3 +
                         1/4*α*io**4 +
                         β*io**5)
            nrg += 0.5 * Co * v[n]**2
        return nrg

    def power(Vin, i, v):
        """
        calculate power dissipated at inputs and outputs and in hotspot
        """
        hspower = 0
        for hotspot in Hotspot.active:  # hotspots first
            hspower += hotspot.history[-1][2]

        if params["RL"] != 0 :  # load resistor power
            term_power = v[-1]**2/params["RL"]
        else:
            term_power = 0

        Rloss = sim_params["rho"]*dx
        hs_indices = Hotspot.active_hotspot_indices()
        if Rloss != 0:
            for n in range(1,len(i)):  # don't count first node, no loss resistor there
                if n not in hs_indices:  # don't count hotspots, no loss resistor in them
                    loss += (v[n] - v[n-1])**2 / Rloss
        else:
            loss = 0
                    
            
        return -i[0]*Vin + \
            params["RIN"]*i[0]**2 + \
            loss + \
            term_power + \
            hspower

    left_boundary = {}
    right_boundary = {}
    left_boundary['type'] = 'source'
    left_boundary['source impedance'] = params['RIN']
    right_boundary['type'] = 'load'
    right_boundary['load impedance'] = params['RL']

    detailed_balance = []

    """
    Central simulation loop
    """
    while t < duration:
        left_boundary["source strength"] = v_IN(t)
        valid_step = False
        energy_before_step = energy(i, v)
        
        power_during_step = power(left_boundary["source strength"],
                                  i,
                                  v)
        while (valid_step != True):
            if args.debug :
                #print(f"Time is {t}")
                pass
            
            # Where simulation occurs
            temp_i,temp_v  = timestep(i,
                                      v,
                                      left_boundary,
                                      right_boundary,
                                      dt)

            energy_after_step = energy(temp_i, temp_v)

            step_nrg_gain = power_during_step*dt + \
                            energy_after_step - \
                            energy_before_step

            # now we have to check that energy is reasonably close to
            # conserved, if not take a smaller timestep.
            nrg_gain_max = sim_params["nrg_gain_max"]
            nrg_gain_min = sim_params["nrg_gain_min"]
            """
            need to check np.abs(energy_after_step) > 0 so we don't get 
            divide by zero error some weird thing with low energy cases. 
            """
            if sim_params["adaptive_time"]:
                if np.abs(energy_after_step) > 1e-25 and \
                   np.abs(step_nrg_gain/energy_after_step) > nrg_gain_max:
                    # moving too quickly
                    valid_step = False
                    # back up hotspot
                    for hotspot in Hotspot.active:
                        hotspot.delete_timestep()

                    dt = dt/2
                    if args.verbose :
                        print(f"Energy change too large, dt = {dt:.3}")
            else :  # not moving too quickly, or not adaptive, or hotspot
                valid_step = True
                i = temp_i
                v = temp_v
                t += dt
                frame_timer += dt
                if sim_params["adaptive_time"] and \
                   np.abs(energy_after_step) > 0 and \
                   np.abs(step_nrg_gain/energy_after_step) < nrg_gain_min:
                    dt = 2*dt
                    if args.verbose :
                        print(f"Energy change too small, dt = {dt:.3}")
                
        # step completed successfully
        detailed_balance.append((t,step_nrg_gain))
        
        if frame_timer > frame_duration:
            frame_timer = 0
            frames.append((t,i,v))
            if args.verbose :
                print(f'{t/duration*100:.3}% completed')

    return frames, detailed_balance

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description =
                                     'Simulate transmission line with nonlinear inductance.')
    parser.add_argument('-f','--filename', help = 'output file name for data dump')  # not implemented FIXME
    parser.add_argument('--mur', default = 1, type = float,
                        help='relative magnetic permeability at zero current')
    parser.add_argument('--epsr', default = 1, type = float,
                        help = 'relative dielectric permeability')
    parser.add_argument('--length', default = 1, type = float,
                        help = 'length of transmission line [m]')
    parser.add_argument('--dx', default = 1e-2, type = float,
                        help = 'distance between nodes [m]')
    parser.add_argument('--dt', default = 1e-5, type = float,
                        help = 'time step [s]')
    parser.add_argument('--duration', default = 1, type = float,
                        help = 'duration of simulation [s]')
    parser.add_argument('--frames', default = 10, type = int,
                        help = 'number of frames to output')
    parser.add_argument('--Rload', default = 'match', 
                        help = 'load impedance [Ω] or "match" to match impedance')
    parser.add_argument('--Rin', default = 'match',
                        help = 'output impedance of source [Ω] or "match" to match impedance')
    parser.add_argument('--alpha', default = '0', type = float,
                        help = 'quadratic term in nonlinearity [A⁻²]')
    parser.add_argument('--beta', default = '0', type = float,
                        help = 'quartic term in nonlinearity [A⁻⁴]')
    parser.add_argument('--source', default = 'gaussian',
                        help = 'source type: gaussian, sinusoid, or step')
    parser.add_argument('--amplitude', default = .001, type = float,
                        help = 'amplitude of source [A]')
    parser.add_argument('--t_offset', default = 1.0, type = float,
                        help = 'time offset from zero of source [s]')
    parser.add_argument('--sigma', default = 0.4, type = float,
                        help = 'standard deviation of gaussian, or period of sinusoidal source [s]')
    parser.add_argument('--offset', default = 0.0, type = float,
                        help = 'DC offset value of source')
    parser.add_argument('-p', '--plot', action = 'store_true',
                        help = 'show plot of output')
    parser.add_argument('-v', '--verbose', action = 'store_true',
                        help = 'verbose output, useful for tracking simulation')
    parser.add_argument('-d', '--debug', action = 'store_true',
                        help = 'debugging mode, for development')
    parser.add_argument('--ic', default = 1, type = float,
                        help = 'switching current of wire')
    parser.add_argument('--ihs', default = .3, type = float,
                        help = 'retrapping current of wire')
    parser.add_argument('--Rsheet', default = 1, type = float,
                        help = 'sheet resistance of film')
    parser.add_argument('--hsloc', default = 1.5, type = float,
                        help = 'constriction location as fraction of length.  If > 1 or < 0, no constriction exists.')
    parser.add_argument('--bias', default = 0, type = float,
                        help = 'initial DC bias current as fraction of ic')
    parser.add_argument('--adaptt', action = 'store_true',
                        help = 'adaptive timestep mode')
    parser.add_argument('--nrgmin', default = 1e-9, type = float,
                        help = 'minimum fractional energy change per timestep')
    parser.add_argument('--nrgmax', default = 1e-9, type = float,
                        help = 'maximum fractional energy change per timestep')
    parser.add_argument('--fcut', default = 40e9, type = float,
                        help = 'maximum frequency in transmission line')
    
    args = parser.parse_args()
    if args.Rin == 'match':
        args.Rin = np.sqrt(args.mur/args.epsr)
    if args.Rload == 'match':
        args.Rload = np.sqrt(args.mur/args.epsr)

    # for parameters that are about the physical system primarily
    params = {"mu_r": args.mur,
              "eps_r": args.epsr,
              "RIN": args.Rin,
              "RL": args.Rload,
              "alpha": args.alpha,
              "beta": args.beta,
              "ic": args.ic,
              "ihs": args.ihs,
              "hotspot_location": args.hsloc,
              "bias": args.bias,
              "length": 1., # always use length 1, and normalize
              # convert duration to sim units by * c/L
              "duration": args.duration * c / args.length,
              "Rsheet": args.Rsheet / Zo
    }

    sourcelist = {'gaussian': gaussian(args.amplitude,
                                       args.t_offset * c / args.length,
                                       args.sigma * c / args.length,
                                       args.offset),
                  'step': step(args.amplitude,
                               args.t_offset * c / args.length,
                               args.sigma * c / args.length,
                               args.offset),
                  'sinusoid': sinusoid(args.amplitude,
                                       args.t_offset * c / args.length,
                                       1/(args.sigma * c / args.length),
                                       args.offset)
    }

    # for parameters that describe the source
    source_params = {"type": args.source,
                     "amplitude": args.amplitude,
                     "t_offset": args.t_offset,
                     "sigma": args.sigma,
                     "offset": args.offset}
    
    # for parameters that are about the numerics primarily
    sim_params = {"dx": args.dx/args.length,
                  "dt": args.dt/args.duration,
                  "numframes": args.frames,
                  "nrg_gain_max" : 1e-6,
                  "nrg_gain_min": 1e-8,
                  "Vretrap": args.ihs * args.Rsheet / Zo * .1,
                  "fcut": args.fcut * args.length / c,
                  "adaptive_time": args.adaptt
                  }
    # resistance per unit length.  all the componetns have already
    # been converted to sim units, so shouldn't require further conversion

    # sim_params["rho"] = 2*np.pi*params["mu_r"]*sim_params["fcut"]
    sim_params["rho"] = 0

    if args.debug:
        print(params)
        print(sim_params)


    frames_out,detailed_balance  = simulate(sourcelist[args.source],
                                            sim_params = sim_params,
                                            params = params)
    
    if args.plot :
        xpoints = [n*args.dx for n in range(int(args.length/args.dx))]
        plt.subplot(2,1,1)
        for frame in frames_out:
            label = f"{frame[0]:.2}"
            temp = plt.plot(xpoints,frame[1],label=label)
        plt.ylabel('current')
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        plt.legend()

        plt.subplot(2,1,2)
        for frame in frames_out:
            plt.plot(xpoints,frame[2])
        plt.xlabel('pos')
        plt.ylabel('voltage')
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        plt.show()

        plt.plot([x[0] for x in detailed_balance], [x[1] for x in detailed_balance])
        plt.xlabel('time')
        plt.ylabel('energy')
        plt.show()

    if args.debug :
        #print(frames_out)
        pass

    if args.filename:
        with open(args.filename, 'w') as f:
            f.write(str(params))
            f.write(str(sim_params))
            f.write(str(source_params))
            # currently the output is not very well enumerated, so you need to use dx, dt,
            # and numframes from preamble to figure out position and time...
            f.write(str(frames_out))
            f.close()
