"""
Nonlinsim

Has fully featured CLI with help.  General advice is timestep should
be much finer than position step (in units where c = 1).

A few examples:

Example 1: runs quickly
python3 nonlinsim.py --mur 1 --epsr 1 --length 8 --dx 1e-1 --dt 1e-3 --duration 8 --frames 10 --Rload 1 --Rin match --alpha .5 --beta .1 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -f test.txt -p

Example 2: for debugging
python nonlinsim.py --mur 1 --epsr 1 --length 1 --dx 1e-1 --dt 1e-3 --duration 1 --frames 10 --Rload 1 --Rin match --alpha 0 --beta 0 --source gaussian --amplitude .001 --phase .5 --sigma .2 --offset 0 -p

Example 3: including hotspot effects
python nonlinsim.py --mur 1 --epsr 1 --length 1 --dx 1e-2 --dt 1e-4 --duration 2 --frames 10 --Rload 1 --Rin 1 --alpha 0 --beta 0 --source step --amplitude 2e-6 --phase .5 --sigma .01 --offset 0 --ic 1e-3 --ihs 0.3e-3 --bias 0 --Rsheet 1 -p -f 2020-01-14-termination1.dat &

Adapative timestep --adaptt If energy loss/gain in system is more than
a fixed fraction of the total energy (say 1%) then throw away that
timestep, divide timestep by 2, and re-run.  If energy loss/gain in
system is less than a fixed fraction of the total energy (say .1%)
then keep that timestep, but set next timestep to be 2x current
timestep.

FIXME refactor out Vretrap and replace with a Rmin

"""
import json
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erf as erf
from scipy import constants,sparse,interpolate

# These are physical constants... they're gonna be globals.

c = constants.speed_of_light
Zo = np.sqrt(constants.mu_0/constants.epsilon_0)

# set up logging
import logging
"""
Hotspot class


"""
class Hotspot:
    # some class attributes that will be accessed as constants
    psi = 38
    vo = 250 / c  # m/s
    f = 0.1

    # some class attributes that will be used by simulation,
    # effectively as globals, but protected in class
    active_hs = set()  # set of all active hotspots
    archive = set()  # set of all past hotspots

    def __init__(self, t_o, ndx, params, sim_params):
        """
        t_o start time (FIXME?)
        ndx:: index
        params:: parameters defining physical systems
        sim_params:: simulation parameters
        """
        self.index = ndx
        self.t_o = t_o
        self.hotspot_age = 0.0
        self.Rs = params['Rsheet']
        self.res = self.f * self.Rs
        self.width = params['width']
        self.V_retrap = sim_params['Vretrap']  # FIXME refactor
        self.isw = params['ic']
        """
        the history vector is the main way we will keep track of the state of the hotspot.
        each tuple will track current time, resistance, and power dissipated in previous timestep
        """
        self.history = np.array([(self.hotspot_age, self.res, 0)])
        Hotspot.active_hs.add(self)
        sim_params["hotspots"][ndx] = 1  # update transmission-line state
        logging.info(f"New hotspot initialized {self}")
        return

    def step(self, i, dt):
        """
        update hotspot state by a time step of duration dt
        
        return True if hotspot has not collapsed
        return False if hotspot has collapsed

        """

        i = i / self.isw
        vhs = 2*self.vo*(self.psi*i**2 - 2) / np.sqrt(self.psi*i**2 -1)
        oldres = self.res
        self.res += vhs * dt * self.Rs / self.width
        self.hotspot_age += dt

        # power dissipated, use average of resistance between timesteps
        dp = i**2 *self.isw**2 * (oldres + self.res)/2.0
        
        if self.res <= self.V_retrap / (np.abs(i)*self.isw):  # hotspot collapsed?
            sim_params["hotspots"][ndx] = 0  # update transmission-line state
            Hotspot.active_hg.remove(self)
            Hotspot.archive.add(self)
        else:
            np.append(self.history, (self.hotspot_age, self.res, dp))
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
        for hotspot in Hotspot.active_hs:
            indices.append(hotspot.index)
        return indices

    """
    get_hotspot

    helper to return hotspot at a given index, or return 'default' (False) if no
    hotspot exists at that index
    """
    @staticmethod
    def get_hotspot(ndx, default = False):
        for hotspot in Hotspot.active_hs:
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
        return amplitude * np.exp(-(t-t_offset)**2/(2*sigma**2)) + offset

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

#@profile
def simulate(sim_params, params):
    """
    simulate for a giving input function and set of parameters

    sim_params :: dict of simulation relevant parameters
    params :: dict of physically relevant parameters

    perform simulation over duration etc. as specified in parameter dictionaries
    """
    length = params["length"]
    duration = params["duration"]

    dx = sim_params["dx"]
    dt = sim_params["dt"]
    numframes = sim_params["numframes"]
    v_IN = params["source"]
    
    try:
        if args.debug:
            times = np.arange(0,duration,dt)
            plt.plot(times,v_IN(times))
            plt.show()
    except:
        pass
    
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

    numpoints = sim_params["numpoints"]

    icnormd = np.ones(numpoints)

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
        i = list(i)
        v = list(v)
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
                try:
                    if args.verbose:
                        print(f"Hotspot detected at n = {n}, newi is {newi[n]:.3}")
                        raise Exception()
                except:
                    pass
                
                # switching current exceeded
                if n not in Hotspot.active_hotspot_indices():
                    # hotspot didn't exist at this location previously, create it
                    Hotspot(t, n, params, sim_params)
            else:
                # check if there's a hotspot here already
                hotspot = Hotspot.get_hotspot(n)
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

                    newi[-1] = dt/mus/dx*dv + i[n]
                    newv[-1] = dt/dx/eps*di + v[n]
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
    
    def make_lookup(func):
        """
        make_lookup::helper function to create a faster table lookup version
        of a function with 1000 points
        """
        ivals = np.linspace(-params["ic"],params["ic"],1000)
        yvals = func(ivals)
        lookup_func = interpolate.interp1d(ivals, yvals)
        try:
            if args.debug:
                plt.plot(ivals,yvals)
                plt.show()
        except:
            pass
        return lookup_func

    @make_lookup
    def mu_eff(current):
        """
        current: current

        specify nonlinear dependence of mu on current.
        """
        return params["mu_r"]*(1
                               + 2*params["alpha"]*current**2
                               + 4*params["beta"]*current**4)
    def faster_timestep(i, v, left_boundary, right_boundary, dt):
        """
        move i,v vectors forward from time t to time t+dt, taking care of
        boundary conditions.

        update any hotspots

        return updated i, v
        """
        # bring in state from outside

        def next_current_vector(i, v, N, dt, dx, μ_func, ε, σ, VIN, RIN, RL):
            μ_vals = μ_func(i)
            i_i = sparse.identity(N, format="csr")
            i_i = np.ones(N)
#            i_i[-1] = -i_i[-1]  # just a guess
            i_i[0] = 1 - RIN*dt/μ_vals[0]/dx
            i_i = sparse.diags(i_i)
            i_v1 = -np.ones(N)*dt/μ_vals/dx
            #FIXME had to change sign of i_v1[0]
            i_v1[0],i_v1[-1] = i_v1[0],i_v1[-1]
            #FIXME had to change sign of i_v2 everywhere
            i_v2=np.ones(N-1)*dt/μ_vals[1:]/dx  # careful about indexing
            i_v2[-1]=i_v2[-1]
            i_v = sparse.diags([i_v1,i_v2],[0,-1], format = "csr")
            inhomog = np.zeros(N)
            inhomog[0] = VIN * dt / μ_vals[0] / dx
            return i_i.dot(i) + i_v.dot(v) + inhomog


        def next_voltage_vector(i, v, N, dt, dx, μ_func, ε, σ, VIN, RIN, RL):
            v_v1 = np.full(N,1+ dt * σ / ε / dx**2)
            v_v1[0] = 1
            # in derived matrix, should be - Rl term, but + term seems to help
            v_v1[-1] =( v_v1[-1] - dt / ε / (RL+.01) / dx)  # fixme kludge
            
            v_v2 = np.full(N-1, dt * σ / ε / dx**2)
            v_v = sparse.diags([v_v1,v_v2],[0,-1], format= "csr")
            #print("v_v\n",v_v.toarray())
            v_i1 = np.full(N,dt/ε/dx)
            v_i2 = np.full(N-1,-dt/ε/dx)
            # fixme changed signs of these, which wasn't believed to be needed, but which
            # helped a great deal
            v_i1[-1],v_i2[-1]=v_i1[-1],v_i2[-1]
            v_i = sparse.diags([v_i1,v_i2], [0,1], format = "csr")
            #print("v_i\n",v_i.toarray())
            return v_i.dot(i) + v_v.dot(v)

        RIN,RL = left_boundary['source impedance'], right_boundary['load impedance']
        numpoints, dx = sim_params["numpoints"], sim_params["dx"]
        εr,σ = params["eps_r"], params["conductivity"]

        nexti = next_current_vector(i, v, numpoints, dt, dx, mu_eff, εr, σ, v_IN(t), RIN, RL)
        nextv = next_voltage_vector(i, v, numpoints, dt, dx, mu_eff, εr, σ, v_IN(t), RIN, RL)

        # check if a hotspot needs to be created
        # fixme create icnormd

        # hotspots is now an array with "True" wherever there's a hotspot
        # hotspots = nexti > params["ic"]*icnormd
        hotspots = np.full(numpoints,False)
        for n in np.argwhere(hotspots):
            if verbose:
                print(f"Hotspot detected at n = {n}, newi is {nexti[n]:.3}")
                raise Exception()  #fixme, breaks hotspots
                pass
            
            # switching current exceeded
            if n not in Hotspot.active_hotspot_indices():
                # hotspot didn't exist at this location previously, create it
                Hotspot(t, n, params, sim_params)
            else:
                # check if there's a hotspot here already
                hotspot = Hotspot.get_hotspot(n)
                if hotspot:
                    Rhs = hotspot.res
                    """
                    
                        v(n-1)       v(n)
                    . . . o--L---Rhs--o
                      ->        |
                      in        C
                                |
                                g
                    
                    """
                    dv = v[n-1] - (v[n] +  i[n] * Rhs)  # voltage across inductor
                    di = i[n] - i[n+1]  # current in capacitor
                    nexti[n] = dt/mu_eff(i[n])/dx*dv + i[n]  # might be very slow b/c i[n] not array
                    nextv[n] = dt/dx/eps*di + v[n]
                    hotspot.step(i[n],dt) == 0

        return nexti, nextv

    def even_faster_timestep(i, v, left_boundary, right_boundary, dt):
        """
        move i,v vectors forward from time t to time t+dt, taking care of
        boundary conditions.

        update any hotspots

        return updated i, v
        """
        # bring in state from outside
        iv = np.array([i,v])
        
        def next_current_vector(i, v, N, dt, dx, μ_func, ε, σ, VIN, RIN, RL):
            μ_vals = μ_func(i)
            i_i = sparse.identity(N, format="csr")
            i_i = np.ones(N)
#            i_i[-1] = -i_i[-1]  # just a guess
            i_i[0] = 1 - RIN*dt/μ_vals[0]/dx
            i_i = sparse.diags(i_i)
            i_v1 = -np.ones(N)*dt/μ_vals/dx
            #FIXME had to change sign of i_v1[0]
            i_v1[0],i_v1[-1] = i_v1[0],i_v1[-1]
            #FIXME had to change sign of i_v2 everywhere
            i_v2=np.ones(N-1)*dt/μ_vals[1:]/dx  # careful about indexing
            i_v2[-1]=i_v2[-1]
            i_v = sparse.diags([i_v1,i_v2],[0,-1], format = "csr")
            inhomog = np.zeros(N)
            inhomog[0] = VIN * dt / μ_vals[0] / dx
            return i_i.dot(i) + i_v.dot(v) + inhomog


        def next_voltage_vector(i, v, N, dt, dx, μ_func, ε, σ, VIN, RIN, RL):
            v_v1 = np.full(N,1+ dt * σ / ε / dx**2)
            v_v1[0] = 1
            # in derived matrix, should be - Rl term, but + term seems to help
            v_v1[-1] =( v_v1[-1] - dt / ε / (RL+.01) / dx)  # fixme kludge
            
            v_v2 = np.full(N-1, dt * σ / ε / dx**2)
            v_v = sparse.diags([v_v1,v_v2],[0,-1], format= "csr")
            #print("v_v\n",v_v.toarray())
            v_i1 = np.full(N,dt/ε/dx)
            v_i2 = np.full(N-1,-dt/ε/dx)
            # fixme changed signs of these, which wasn't believed to be needed, but which
            # helped a great deal
            v_i1[-1],v_i2[-1]=v_i1[-1],v_i2[-1]
            v_i = sparse.diags([v_i1,v_i2], [0,1], format = "csr")
            #print("v_i\n",v_i.toarray())
            return v_i.dot(i) + v_v.dot(v)

        def update_i_v(iv, N, dt, dx, μ_func, ε, σ, VIN, RIN, RL):
            i,v = iv[0],iv[1]
            nexti = next_current_vector(i, v, numpoints, dt, dx, mu_eff, εr, σ, v_IN(t), RIN, RL)
            nextv = next_voltage_vector(i, v, numpoints, dt, dx, mu_eff, εr, σ, v_IN(t), RIN, RL)
            iv = np.array([nexti,nextv])
            return iv
            
        RIN,RL = left_boundary['source impedance'], right_boundary['load impedance']
        numpoints, dx = sim_params["numpoints"], sim_params["dx"]
        εr,σ = params["eps_r"], params["conductivity"]

        i,v = update_i_v(iv, numpoints, dt, dx, mu_eff, εr, σ, v_IN(t), RIN, RL)
        return i, v
#        return iv[0], iv[1]

    def adaptive_timestep(i, v, left_boundary, right_boundary, dt, params=params, sim_params=sim_params):
        """
        adaptive_timestep::move i,v vectors forward from time t to time t+dt making sure energy limits
                           are not exceeded, modifying dt accordingly.  Note, this runs recursively to
                           build up the result.  Do not run with excessively large dt, or stack might blow up.
        
        result::array of (i,v,dt) tuples representing steps that have occurred since first invocation

        """
        nonlocal t
        result = []
        newi,newv = faster_timestep(i, v, left_boundary, right_boundary, dt)

        starting_nrg = energy(i,v)
        nrg_diss = power(params["source"],i,v, params, sim_params)*dt
        Δnrg = np.abs(starting_nrg - energy(newi, newv) - nrg_diss)/(starting_nrg + nrg_diss + 1e-30)

        max_Δnrg = sim_params["max_Δnrg"]*starting_nrg
        if Δnrg > max_Δnrg :
            logging.info("max Δnrg exceeded, digging down")
            for hotspot in Hotspot.active_hs:
                hotspot.delete_timestep()
            newi,newv = adaptive_timestep(i, v, left_boundary, right_boundary, dt/2)
            t+=dt/2
            newi,newv = adaptive_timestep(newi, newv, left_boundary, right_boundary, dt/2)

        return newi, newv

    def energy(i,v):
        """
        faster version of
        calculate total energy stored in transmission line
        """
        i,v = np.array(i), np.array(v)
        Co = params["eps_r"] * dx
        nrg = 0
        Lo = params["mu_r"] * dx
        α = params["alpha"]
        β = params["beta"]
        # identified as key source of slow-down, so use numpy
        node_i_nrg = Lo * (0.5*i**2 +
                           2/3*α*i**3 +
                           1/4*α*i**4 +
                           β*i**5)
        node_v_nrg = 0.5 * Co * v**2

        return np.sum(node_i_nrg + node_v_nrg)

    def power(Vin, i, v, params, sim_params):
        """
        calculate power dissipated at inputs and outputs and in hotspot
        """
        i,v = np.array(i),np.array(v)
        RL = params["RL"]
        hspower = 0
        for hotspot in Hotspot.active_hs:  # hotspots first
            hspower += hotspot.history[-1][2]

        # load resistor power
        if RL != 0 :
            term_power = v[-1]**2/RL
        else:
            term_power = 0

        Rloss = sim_params["rho"]*dx
        if Rloss:
            loss_vals = np.diff(v)
            loss_vals *= (1-sim_params["hotspots"])  # zero out hotspot locations
            loss_vals *= loss_vals/Rloss  # take v^2/Rloss
            loss = np.sum(loss_vals)
        else:
            loss = 0
            
        return -i[0]*Vin(t) + \
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
    v = np.full(numpoints,params["bias"]*params["ic"]*params["RL"])
    i = np.full(numpoints,params["bias"]*params["ic"])

    while t < duration:
        left_boundary["source strength"] = v_IN(t)
        if sim_params["adaptive_time"]:
            print("in adaptive timestep")
            temp_i, temp_v  = adaptive_timestep(i,
                                                v,
                                                left_boundary,
                                                right_boundary,
                                                dt)
            i,v = temp_i, temp_v
            t += dt
            frame_timer += dt
            
        else :  # not moving too quickly, or not adaptive, or hotspot
#            i, v  = timestep(old_i,
#                                     old_v,
#                                     left_boundary,
#                                     right_boundary,
#                                     dt)
#            i,v  = adaptive_timestep(i,
#                                   v,
#                                   left_boundary,
#                                   right_boundary,
#                                   dt)
            i,v  = faster_timestep(i,
                                   v,
                                   left_boundary,
                                   right_boundary,
                                   dt)
            #print("old_i,old_v\n", old_i, old_v)
            #print("i,v\n", i,v)
            #print("diff",old_i - i, old_v-v)
            t += dt
            frame_timer += dt
            
        # step completed successfully
        
        if frame_timer > frame_duration:
            frame_timer = 0
            frames.append((t,i,v))
            try:
                if args.verbose :
                    print(f'{t/duration*100:.3}% completed')
                logging.info(f'{t/duration*100:.3}% completed')
            except:
                pass
#            print([frames[-1]])
#            print([(t,old_i,old_v)])
#            compare_frames([frames[-1]],[(t,old_i,np.array(old_v))],params,sim_params)
#            try:
#                if args.debug :
#
#            except:
#                pass
            
    return frames, detailed_balance

import argparse

parser = argparse.ArgumentParser(description =
                                 'Simulate transmission line with nonlinear inductance.')
parser.add_argument('-f','--filename', type = str, help = 'output file name for data dump')
parser.add_argument('--mur', default = 1, type = float,
                    help='relative magnetic permeability at zero current')
parser.add_argument('--epsr', default = 1, type = float,
                    help = 'relative dielectric permeability')
parser.add_argument('--length', default = 1, type = float,
                    help = 'length of transmission line [m]')
parser.add_argument('--Nx', default = 100, type = int,
                    help = 'number of x points')
parser.add_argument('--dt', default = 1e-5, type = float,
                    help = 'time step [s]')
parser.add_argument('--duration', type = float,
                    help = 'duration of simulation [s]')
parser.add_argument('--frames', default = 10, type = int,
                    help = 'number of frames to output')
parser.add_argument('--Rload',
                    help = 'load impedance [Ω].  If absent, impedence assumed to be matched')
parser.add_argument('--Rin',
                    help = 'output impedance of source [Ω]. If absent, assumed to be matched')
parser.add_argument('--alpha', default = '0', type = float,
                    help = 'quadratic term in nonlinearity [1/A**2]')
parser.add_argument('--beta', default = '0', type = float,
                    help = 'quartic term in nonlinearity [1/A**4]')
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
parser.add_argument('--freq', default = 0.0, type = float,
                    help = 'frequency in Hz')
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
parser.add_argument('--width', default = 100.0e-9, type = float,
                    help = 'width of nanowire [m]')

def text_to_args(arg_str):
    """take raw text and convert it to a list of args for parsing as if on command line"""
    args = parser.parse_args(arg_str.split())
    return args
    
   
def cla_to_dicts(args):
    """
    cla_to_dicts::convert human-comfortable version of inputs to units and format
    that the simulator uses
    """
        
    # set Rin and Rload to match if they weren't defined
    if args.Rin == None:
        args.Rin = np.sqrt(args.mur/args.epsr)*Zo
        if args.verbose:
            print("No Rin entered, using matched impedence condition")

    if args.Rload == None:
        if args.verbose:
            print("No Rload entered, using matched impedence condition")
        args.Rload = np.sqrt(args.mur/args.epsr)*Zo

    """
    if we change our timescale to units of distance, we can eliminate
    multiplications by small numbers like μo and εo.  We can do this by
    multiplying all our times by the speed of light.
    """
    transit_time = args.length / c * np.sqrt(args.mur*args.epsr)
    # inf duration not specified, simulate across one transit time

    # for parameters that are about the physical system primarily
    params = {"mu_r": args.mur,
              "eps_r": args.epsr,
              "RIN": args.Rin/Zo,
              "RL": args.Rload/Zo,
              "alpha": args.alpha,
              "beta": args.beta,
              "ic": args.ic,
              "ihs": args.ihs,
              "hotspot_location": args.hsloc,
              "bias": args.bias,
              "length": args.length,
              "duration": args.duration * c,  # distance units
              "Rsheet": args.Rsheet / Zo,
              "width": args.width/args.length  # fixme, weird
    }

    sourcelist = {'gaussian': gaussian(args.amplitude/Zo,
                                       args.t_offset*c,
                                       args.sigma*c,
                                       args.offset/Zo),
                  'step': step(args.amplitude/Zo,
                               args.t_offset*c,
                               args.sigma*c,
                               args.offset/Zo),
                  'sinusoid': sinusoid(args.amplitude/Zo,
                                       args.freq/c,
                                       args.t_offset*c,
                                       args.offset/Zo)
    }

    # for parameters that describe the source
    source_params = {"type": args.source,
                     "amplitude": args.amplitude/Zo,
                     "t_offset": args.t_offset / transit_time,
                     "sigma": args.sigma / transit_time,
                     "offset": args.offset/Zo}
    
    # for parameters that are about the numerics primarily
    sim_params = {"dx": args.length/args.Nx,
                  "dt": args.dt*c,
                  "numframes": args.frames,
                  "max_Δnrg" : 1e-6,
                  "nrg_Δnrg": 1e-8,
                  "Vretrap": args.ihs * args.Rsheet / Zo * .1,
                  "fcut": args.fcut / c,  # note units are 1/dist
                  "adaptive_time": args.adaptt,
                  "plot": args.plot,
                  "file": args.filename,
                  }
    # resistance per unit length.  all the componetns have already
    # been converted to sim units, so shouldn't require further conversion

    # sim_params["rho"] = 2*np.pi*params["mu_r"]*sim_params["fcut"]
    sim_params["rho"] = 0
    params["conductivity"] = 0
    if args.debug:
        print(f"got here {args.debug}")
        print(f"transmission line length {params['length']}")
        print(f"speed of light {np.sqrt(params['mu_r']/params['eps_r'])}c")
        print(f"Time params:")
        print(f"input dt: {args.dt}")
        print(f'timescale (L/c_eff): {transit_time}')
        print(f'sim dt: {sim_params["dt"]}')
        print(f'input duration: {args.duration}')
        print(f'sim duration: {params["duration"]}')
        print(f'num timesteps: {int(params["duration"]/sim_params["dt"])}')

    
    if args.debug:
        print(params)
        print(sim_params)

    params["source"] = sourcelist[args.source]
    sim_params["numpoints"] = int(params["length"]/sim_params["dx"])

    # we need an array that tracks the hotspots in the 
    sim_params["hotspots"] = np.zeros(sim_params["numpoints"])

    # set up logging
    configure_logs(args.verbose, args.debug)
    
    return params, sim_params, source_params

def plot_frames(frames_out, params, sim_params):
    dx = sim_params["dx"]
    xpoints = [dx*n for n in range(int(params["length"]/dx))]
    plt.subplot(2,1,1)
    for frame in frames_out:
        label = f"{frame[0]:.2}"
        temp = plt.plot(xpoints,frame[1],label=label)
    plt.ylabel('current')
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    plt.legend()

    plt.subplot(2,1,2)
    for frame in frames_out:
        plt.plot(xpoints,np.array(frame[2])*Zo)
    plt.xlabel('pos')
    plt.ylabel('voltage')
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    plt.show()

def compare_frames(frames_out1, frames_out2, params, sim_params):
    xpoints = np.arange(0,params["length"],sim_params["dx"])
    plt.subplot(2,1,1)
    for frame1,frame2 in zip(frames_out1,frames_out2):
        t1,i1,v1 = frame1
        t2,i2,v2 = frame2
        plt.plot(xpoints,i1)
        plt.plot(xpoints,i2)
    plt.ylabel('current')
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    plt.legend()

    plt.subplot(2,1,2)
    for frame1,frame2 in zip(frames_out1,frames_out2):
        t1,i1,v1 = frame1
        t2,i2,v2 = frame2
        plt.plot(xpoints,v1*Zo)
        plt.plot(xpoints,v2*Zo)
    plt.xlabel('pos')
    plt.ylabel('voltage')
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    plt.show()

def plot_detailed_balance(detailed_balance):
    plt.plot([x[0] for x in detailed_balance], [x[1] for x in detailed_balance])
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.show()

def save_frames(frames_out, filename):
    with open(filename+'.json', 'w') as f:
        json.dump(frames_out,f)

def read_frame(filename):
    with open(filename+'.json', 'r') as f:
        frames = json.load(f)
    return frames

def configure_logs(verbose, debug):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    if debug:
        logging.basicConfig(level=logging.DEBUG)

import unittest

import time
def test1():
    args = text_to_args("""
    --mur 100 --epsr 1 --length .01 --Nx 10 --dt 1e-13 --duration 1.5e-9 --frames 10 --alpha 4e4 --beta 4e8 
    --source gaussian --amplitude .001 --ic 1.2e-3 --t_offset 1e-9 --sigma 3e-10 --offset 0


    """)
    params,sim_params,source_params = cla_to_dicts(args)
    start_time = time.time()
    frames_out,detailed_balance  = simulate(sim_params, params)
    print(f"Test1 ran in {time.time() - start_time} seconds")
    for frame1, frame2 in zip(frames_out, read_frame("test1")):
        t1,i1,v1 = frame1
        t2,i2,v2 = frame2
        assert np.all(np.isclose(i1,i2)), "Test 1 failed to match expectation"
    print("Test 1 passed")
    return 0

def test2():
    args = text_to_args("""
    --mur 100 --epsr 1 --length .1 --Nx 100 --dt 1e-12 --duration 2e-9 --frames 10 --alpha 4e4 --beta 4e8 
    --source step --amplitude .377 --ic 1.2e-3 --t_offset 5e-10 --sigma 1.5e-10 --offset 0
    """)
    params,sim_params,source_params = cla_to_dicts(args)
    start_time = time.time()
    frames_out,detailed_balance  = simulate(sim_params, params)
    print(f"Test2 ran in {time.time() - start_time} seconds")
#    for frame1, frame2 in zip(frames_out, read_frame("test1")):
#        t1,i1,v1 = frame1
#        t2,i2,v2 = frame2
#        assert np.all(np.isclose(i1,i2)), "Test 1 failed to match expectation"
#    print("Test 1 passed")
    return 0

def step_resistor(hotspots, i, dt, params, sim_params):
        """
        update hotspot states by a time step of duration dt
        
        return True if hotspot has not collapsed
        return False if hotspot has collapsed

        """
        
        isw = 0.8 * params["ic"]
        i /= isw
        vhs = 2*self.vo*(self.psi*i**2 - 2) / np.sqrt(self.psi*i**2 -1)
        oldres = self.res
        self.res += vhs * dt * self.Rs / self.width
        self.hotspot_age += dt

        # power dissipated, use average of resistance between timesteps
        dp = i**2 *self.isw**2 * (oldres + self.res)/2.0
        
        if self.res <= self.V_retrap / (np.abs(i)*self.isw):  # hotspot collapsed?
            sim_params["hotspots"][ndx] = 0  # update transmission-line state
            Hotspot.active_hg.remove(self)
            Hotspot.archive.add(self)
        else:
            np.append(self.history, (self.hotspot_age, self.res, dp))
        return

    
if __name__=='__main__':
    #frames_out,detailed_balance  = simulate(sim_params = sim_params,params = params)
    test1()
    test2()
    
#    first_test = TestCase()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        params,sim_params,source_params = cla_to_dicts(args)
        frames_out,detailed_balance  = simulate(sim_params = sim_params,
                                                params = params)
        if sim_params["plot"] :
            plot_frames(frames_out, params, sim_params)

        if sim_params["file"] :
            save_frames(frames_out, params, sim_params, source_params)
    # if no arguments provided on command line, use examle hard-coded params
    else:
        args = text_to_args("""
    --length 1 --mur 1 --epsr 1 --Nx 100 --dt 1e-12 --frames 10 --alpha 0
    --beta 0 --source step --amplitude .0001 --t_offset 6e-9 --sigma 1e-9 
    --offset 0.0 --ic 1.2e-3 --ihs 0.4e-3 --bias 0 --Rsheet 400 --duration 2e-8 
    -v -p""")

