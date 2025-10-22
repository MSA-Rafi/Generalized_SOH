
def warp(voltage, current, time_stamps, ref_min, ref_max, target_len, del_t_ref, start, step, rated_cap):
#     #omitting the voltage sample >ref_max and <ref_min
#     start_index = np.where(voltage > ref_max)[0]
#     if start_index.size == 0: start_index = [-1]
#     start_index = start_index[-1] + 1
    
#     end_index = np.where(voltage < ref_min)[0]
#     if end_index.size == 0: end_index = [voltage.size]
#     end_index = end_index[0]
    
#     voltage = (voltage[start_index:end_index] - ref_min) / (ref_max - ref_min) #crop and normalize
#     current = current[start_index:end_index] #crop current accordingly  
#     time_stamps = time_stamps[start_index:end_index] #crop time stamps accordingly
    
    voltage = voltage / ref_max
    
    sampling_indices = np.arange(time_stamps[0], time_stamps[-1] , del_t_ref)
    voltage_interp_func1 = interp1d(time_stamps, voltage, kind='cubic')
    voltage_sampled = voltage_interp_func1(sampling_indices)
    
    current_interp_func1 = interp1d(time_stamps, current, kind='cubic')
    current_sampled = current_interp_func1(sampling_indices)
    
    target_energy = find_energy(voltage_sampled, current_sampled, del_t_ref)
    energy_tolerance = target_energy / 10000
    
    sampling_indices = np.arange(0,len(voltage_sampled))
    voltage_interp_func2 = interp1d(sampling_indices, voltage_sampled, kind='cubic')
    voltage_interp = voltage_interp_func2(np.linspace(sampling_indices[0], sampling_indices[-1], target_len))
#     del_t = (sampling_indices[-1]-sampling_indices[0])/(target_len-1)
    

    A = 1 / voltage[0]  
    ref_current = rated_cap / 1 #1 hour
   
    a = start
    win = A * window(a)
    voltage_modified = voltage_interp * win
    prev_energy = find_energy(voltage_modified,ref_current,del_t_ref)
#     print("in fnc")
    while True:
        a = a + step
        win = A * window(a)
        voltage_modified = voltage_interp * win
        new_energy = find_energy(voltage_modified,ref_current,del_t_ref)

        if np.abs(new_energy - target_energy) < energy_tolerance:
            return voltage_modified

        if new_energy > target_energy: 
            a = a - step
            step /= 2
        else:
            prev_energy = new_energy

def find_energy(voltage,current, del_t):
    return np.abs(np.sum(voltage*current*del_t)) 

def window(a, n=0.9):
    if a>1:  a=1-0.001; #print(f"error: a={a}")
    length = 360 - 1
    win_fn = lambda x, a, n: (1-a) * np.exp(-(x**n) / (100*a)) + a
    x = np.arange(length + 1)
    win_1 = win_fn(x, a, n)
    win_2 = - win_fn(length-x, 1-a, n) + 1
    win = win_1 * (length - x)/length  +  win_2 * x/length
    # adding another func to boost energy
    shift = 180
    boost = np.exp(-(np.abs(x-shift)*2/length)**3) * (1 - (np.abs(x-length/2)*2/length)**4)  *     (a-0.9)*10        if a>0.9 else 0
    #                      func to add                              varing weights          effect to main window
    return win + boost



def warp_cycles(voltages_dis, currents_dis, time_stamps_dis, ref_min, ref_max, rated_cap):
    ref_cycle_len = 360
    ref_sampling_int = 10 #seconds
    no_of_cycles = len(voltages_dis)

    warped_voltages = np.zeros((no_of_cycles, ref_cycle_len))

    for cycle_no in range(no_of_cycles):
        # print("In Cycle: ", cycle_no)
        v_ = voltages_dis[cycle_no] #, 0:cycle_lengths[cycle_no]]
        c_ = currents_dis[cycle_no]
        t_ = time_stamps_dis[cycle_no] #, 0:cycle_lengths[cycle_no]]

        warped_v = warp(v_, c_, t_, ref_min, ref_max, ref_cycle_len, ref_sampling_int, 0.001, 1, rated_cap)

        warped_voltages[cycle_no, :] = (warped_v)

    # Now, 'warped_voltages' contains the result for each cycle
    print(warped_voltages.shape)
    plt.figure(figsize=(10,5))
    plt.plot(warped_voltages.T)
    plt.title("Voltae after warping")
    plt.show()
    
    return warped_voltages


def interp_cycles(x, t):
    
    ref_cycle_len = 360
    ref_sampling_int = 10 #seconds
    no_of_cycles = len(x)
    
    interp_x = np.zeros((no_of_cycles, ref_cycle_len))
    
    for i in range(no_of_cycles):
        time_stamps = t[i]
        #time_stamps = time_stamps-time_stamps[0]
        
        sampling_indices = np.arange(time_stamps[0], time_stamps[-1], 10)
        interp_func = interp1d(time_stamps, x[i], kind='cubic')
        sampled_x = interp_func(sampling_indices)
        
        sampling_indices = np.arange(0, len(sampled_x))
        interp_func2 = interp1d(sampling_indices, sampled_x, kind='cubic')
        interp_x[i,:] = interp_func2(np.linspace(sampling_indices[0], sampling_indices[-1], ref_cycle_len))
    
    print(interp_x.shape)
    plt.figure(figsize=(10,5))
    plt.plot(interp_x.T)
    plt.title("Current/Temp after interpolation")
    plt.show()
    
    return interp_x
