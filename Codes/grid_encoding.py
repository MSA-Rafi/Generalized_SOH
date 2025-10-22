import scipy.io
from scipy.interpolate import interp1d
def process_data2(loaded_data, min_length, Max_Charge_Voltage, Discharge_Cutoff, max_cap):
    L,Li, Lt = 100,100,100
    Voltage = loaded_data['voltages']
    Current = loaded_data['currents']
    Temp = loaded_data['temperatures']
        
    # Access the loaded variables
#     loaded_all_data = loaded_data['all_data']
#     loaded_all_data = np.transpose(loaded_all_data, (2, 1, 0))

    all_encoded_data= np.zeros((Voltage.shape[0],min_length, L+Li+Lt))
    loaded_new_capacity = loaded_data['capacitys']
#     if 'Capacity' in loaded_data:
#        loaded_new_capacity = loaded_data['capacitys']
#     else:
#        loaded_new_capacity = loaded_data['capacity']

    labels = np.abs(loaded_new_capacity)/max_cap


    # Iterate over each element along the first dimension
    for i in range(Voltage.shape[0]):
        #nonzero_V_indices = np.nonzero(loaded_all_data[i, :, 0])
        V = Voltage[i,:]
        I = Current[i,:]
        T = Temp[i,:]
        


        truncated_v = V
        Vmax, Vmin = Max_Charge_Voltage, Discharge_Cutoff
        grid_boundaries = np.linspace(Vmin, Vmax, L + 1)
        encoded_voltages = ((truncated_v[:, None] >= grid_boundaries[:-1]) & (truncated_v[:, None] < grid_boundaries[1:])).astype(float)
        encoded_voltages[:, -1] = (encoded_voltages[:, -1] == 1) | (truncated_v == Vmax)
        
        truncated_i = I
        truncated_i = np.abs(truncated_i)/max_cap
        Imax, Imin = 20, 0
        grid_boundaries = np.linspace(Imin, Imax, Li + 1)
        encoded_currents = ((truncated_i[:, None] >= grid_boundaries[:-1]) & (truncated_i[:, None] < grid_boundaries[1:])).astype(float)
        encoded_currents[:, -1] = (encoded_currents[:, -1] == 1) | (truncated_i == Imax)


        truncated_T = T
        Tmax, Tmin = 50, 1
        grid_boundaries = np.linspace(Tmin, Tmax, Lt + 1)
        encoded_temps = ((truncated_T[:, None] >= grid_boundaries[:-1]) & (truncated_T[:, None] < grid_boundaries[1:])).astype(float)
        encoded_temps[:, -1] = (encoded_temps[:, -1] == 1) | (truncated_T == Tmax)
        # Encode the data into grids
        

        all_encoded_data[i,:,:]= np.concatenate((encoded_voltages,encoded_currents,  encoded_temps), axis=1)#, encoded_temps




    return all_encoded_data,labels
