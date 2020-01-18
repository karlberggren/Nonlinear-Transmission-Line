"""
Nonlinsim

2020-01-03 fix current calculation to get inductance right
2020-01-02 Check energy balance

Has fully featured CLI with help.  General advice is timestep should
be much finer than position step (in units where c = 1).

Example 1: runs quickly
python3 nonlinsim.py --mur 1 --epsr 1 --length 8 --dx 1e-1 --dt 1e-3 --duration 8 --frames 10 --Rload 1 --Rin match --alpha .5 --beta .1 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -f test.txt -p

Example 2: for debugging
python3 nonlinsim.py --mur 1 --epsr 1 --length 1 --dx 1e-1 --dt 1e-3 --duration 1 --frames 10 --Rload 1 --Rin match --alpha 0 --beta 0 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -p

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as erf

constants = {"mu_o": 1,
             "eps_o": 1}
params = {"mu_r": 1,
          "eps_r": 1,
          "RIN": 1,
          "RL": 1,
          "alpha": .5,
          "beta": .1}

"""
Create some helper functions, to make it easier to create some generic
inputs.  Of course, user can always make themselves a custom function
to pass as input.
"""

def gaussian(amplitude, center, sigma, offset = 0):
    """
    generic gaussian function, for creating gaussian input pulse
    """
    def v_IN(t):
        return amplitude * np.exp(-(t-center)**2/(2*sigma**2)) + offset

    return v_IN

def sinusoid(amplitude, frequency, phase = 0, offset = 0):
    """
    generic sinusoidal function, for creating sinusoidal input
    """
    def v_IN(t):
        return amplitude * np.cos(2*np.pi*frequency + phase) + offset

    return v_IN

def step(amplitude, time, width, offset):
    """
    generic sigmoid function
    """
    def v_IN(t):
        return amplitude*(1/2+1/2*erf((t-time)/width)) + offset

    return v_IN

def funcsum(func1, func2):
    """
    sum two of our other functions.  To use this, first create the functions
    using step, sinusoid, or gaussian, then pass those two functions to funcsum
    and it will return a function that is the sum.  No CLI yet.  Better for GUI.
    """
    def v_IN(t):
        return func1(t) + func2(t)

    return v_IN
    
def simulate(v_IN, length=100e-6, dx=1e-6, duration=1e-10, dt=1e-12,
             numframes=10, constants = constants, params = params):
    
    def mu(current):
        """
        current: current

        specify nonlinear dependence of mu on current
        """
        mu_r = params["mu_r"]*(1 + params["alpha"]*current**2 +
                               params["beta"]*current**4)
        return  mu_r * constants["mu_o"]

    def mu_eff(current):
        """
        it turns out the i dL/dt effect on voltage can be folded in by
        using an "effective mu" (see notes of 2020-01-02).
        """
        mu_eff = params["mu_r"] * \
                 constants["mu_o"] *(1 +
                                     3*params["alpha"]*current**2 +
                                     5*params["beta"]*current**4)
        return mu_eff
                                                      

    timesteps = int(duration/dt)  # total number of steps
    times = [n*dt for n in range(timesteps)]
    """
    The "frames" and "framestep" stuff is just so that we can sample the
    data along the way, to create animations or visualize the progression of
    the signal in time.  Nothing related to the physics.
    """

    framestep = int(timesteps/numframes)  # number of steps per frame
    frames = []  # array for storing frames as they are produced
    newframe = [(n+1)%framestep for n in range(timesteps)]

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

    # initialize system with no hotspots
    
    hotspot = [False for _ in range(numpoints)]
    
    """
    Now we'll iterate through the timesteps.
    """

    def timestep(i, v, hotspot,left_boundary,right_boundary):
        """
        move i,v vector forward from time t to time t+dt, taking care of boundary conditions.
        """
        newi = []  # i(t+dt)
        newv = []  # v(t+dt)
        newhotspot = hotspot[:]
        
        if args.debug:
            pass
        
        eps = params["eps_r"]*constants["eps_o"]

        # deal with left boundary
        if left_boundary['type'] == 'source':
            mus = mu_eff(i[0])
            Vin = left_boundary['source strength']
            RIN = left_boundary['source impedance']
            newi.append(dt / mus / dx * (Vin - v[0]) + i[0]*(1 - dt * RIN / mus/dx))
            newv.append(dt/dx/eps*i[0] + v[0] - dt/dx/eps*i[1])

        # deal with main body of array

        for n in range(1,len(i)-1):  # avoid endpoints
            mus = mu_eff(i[n])
            # first do routine calculations, will overwrite if needed
            newi.append(dt/mus/dx*v[n-1] + i[n] - dt/mus/dx*v[n])
            newv.append(dt/dx/eps*i[n] + v[n] - dt/dx/eps*i[n+1])
            if hotspot[n] == False:
                if abs(newi[n]) > params["ic"]*icnormd[n]:
                    if args.debug:
                        print(newi[n], params["ic"]*icnormd[n])
                        print('hotspot found got here')
                        pass
                    
                    newhotspot[n] = True
                    if args.debug:
                        print("hot spot formed")
                        pass
                    
                    newi[n] = np.sign(newi[n])*params["ihs"]
            else:  # check for healing
                old_voltage = v[n]-v[n-1]
                new_voltage = newv[n] - newv[n-1]
                if (abs(new_voltage) < params["Vretrap"]) or (np.sign(new_voltage) != np.sign(old_voltage)):
                    # healed
                    newhotspot[n] = False
                    if args.debug:
                        print("hotspot healed")
                        pass
                    
                else:  # not healed, overwrite previous calc
                    newi[n] = i[n]
                    
        # deal with right boundary
        if right_boundary['type'] == 'load':
            Rload = right_boundary['load impedance']
            # tried on Jan 14 2020 to improve this termination condition
            newv.append(v[-1]*(1 - dt / (Rload * eps * dx)) + i[-1] * dt / eps / dx)
            newi.append(dt/mus/dx*v[-2] + i[-1] - dt/mus/dx*v[-1])
            
        if args.debug :
            print(sum(newhotspot))
            pass
        return newi,newv,newhotspot

    def energy(i,v,hotspot) :
        """
        calculate total energy stored in transmission line
        """
        eps = params["eps_r"]*constants["eps_o"]
        nrg = 0
        Lo = constants["mu_o"]*params["mu_r"]*dx
        α = params["alpha"]
        β = params["beta"]
        for n in range(0,len(i)):
            if hotspot[n] is not True :
                io = i[n]
                nrg += Lo*(0.5*io**2 + 2/3*α*io**3 + 1/4*α*io**4 + β*io**5)
            nrg += 0.5 * eps * dx * v[n]**2
            """ I have a small concern, which is I sum as if there is a cap
            at the last node, but we actually throw it out... circuit isn't "real"
            """
        return nrg

    def power(Vin, i, v, hotspot) :
        """
        calculate power dissipated at inputs and outputs and in hotspot
        """
        #print(hotspot)
        if True in hotspot:
            hsndx = hotspot.index(True)
            hspower = (v[hsndx]-v[hsndx-1])*i[hsndx]
        else:
            hspower = 0

        return -i[0]*Vin + params["RIN"]*i[0]**2 + params["RL"]*i[-1]**2 + hspower

    left_boundary = {}
    right_boundary = {}
    left_boundary['type'] = 'source'
    left_boundary['source impedance'] = params['RIN']
    right_boundary['type'] = 'load'
    #right_boundary['type'] = 'hotspot'
    right_boundary['load impedance'] = params['RL']
    #right_boundary['current'] = .001

    detailed_balance = []
    
    for t,newframeval in zip(times,newframe):
        left_boundary["source strength"] = v_IN(t)
        if args.debug:
            #print(f"length of array is {len(i)},{len(v)}.")
            pass
        energy_before_step = energy(i,v,hotspot)
        power_during_step = power(left_boundary["source strength"], i, v, hotspot)
        i,v,hotspot = timestep(i, v, hotspot, left_boundary, right_boundary)
        if args.debug:
            print(sum(hotspot))
            pass
        energy_after_step = energy(i,v,hotspot)
        #print(t,power_during_step, energy_before_step, energy_after_step)
        detailed_balance.append((t,(power_during_step*dt - energy_before_step + energy_after_step)))
        #detailed_balance.append((t,power_during_step))
        
        if args.debug:
            if sum(hotspot) > 0:
                print(t, sum(hotspot))
        if newframeval == 0 :
            frames.append((i,v))
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
                        help = 'load impedance [ohms] or "match" to match impedance')
    parser.add_argument('--Rin', default = 'match',
                        help = 'output impedance of source [ohms] or "match" to match impedance')
    parser.add_argument('--alpha', default = '0', type = float,
                        help = 'quadratic term in nonlinearity [A**(-2)]')
    parser.add_argument('--beta', default = '0', type = float,
                        help = 'quartic term in nonlinearity [A**(-4)]')
    parser.add_argument('--source', default = 'gaussian',
                        help = 'source type: gaussian, sinusoid, or step')
    parser.add_argument('--amplitude', default = .001, type = float,
                        help = 'amplitude of source')
    parser.add_argument('--phase', default = 1.0, type = float,
                        help = 'time offset from zero of source')
    parser.add_argument('--sigma', default = 0.4, type = float,
                        help = 'standard deviation of gaussian, or period of sinusoidal source')
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
    
    args = parser.parse_args()
    if args.Rin == 'match':
        args.Rin = np.sqrt(args.mur*constants['mu_o']/args.epsr/constants['eps_o'])
    if args.Rload == 'match':
        args.Rload = np.sqrt(args.mur*constants['mu_o']/args.epsr/constants['eps_o'])

    params = {"mu_r": args.mur,
              "eps_r": args.epsr,
              "RIN": float(args.Rin),
              "RL": float(args.Rload),
              "alpha": args.alpha,
              "beta": args.beta,
              "ic": args.ic,
              "ihs": args.ihs,
              "Vretrap": args.ihs*args.Rsheet*.1,
              "hotspot_location": args.hsloc,
              "bias": args.bias
              
    }

    sim_params = {"length": args.length,
                  "dx": args.dx,
                  "duration": args.duration,
                  "dt": args.dt,
                  "numframes": args.frames}

    source_params = {"type": args.source,
                     "amplitude": args.amplitude,
                     "phase": args.phase,
                     "sigma": args.sigma,
                     "offset": args.offset}
    
    sourcelist = {'gaussian': gaussian(args.amplitude,
                                       args.phase,
                                       args.sigma,
                                       args.offset),
                  'step': step(args.amplitude,
                               args.phase,
                               args.sigma,
                               args.offset),
                  'sinusoid': sinusoid(args.amplitude,
                                       args.phase,
                                       1/args.sigma,
                                       args.offset)
                  }

    frames_out,detailed_balance  = simulate(sourcelist[args.source],
                                            length = args.length,
                                            dx = args.dx,
                                            duration = args.duration,
                                            dt = args.dt,
                                            numframes = args.frames,
                                            params = params)
    
    if args.plot :
        plt.subplot(2,1,1)
        for frame in frames_out:
            plt.plot(frame[0])
        plt.ylabel('current')
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))

        plt.subplot(2,1,2)
        for frame in frames_out:
            plt.plot(frame[1])
        plt.xlabel('pos')
        plt.ylabel('voltage')
        plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        plt.show()

        #plt.plot([x[0] for x in detailed_balance],[x[1] for x in detailed_balance])
        plt.plot([x[1] for x in detailed_balance])
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
            # FIXME:
            # currently the output is not very well enumerated, so you need to use dx, dt, and numframes from preamble to
            # figure out position and time...
            f.write(str(frames_out))
            f.close()
