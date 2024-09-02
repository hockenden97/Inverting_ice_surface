# This module contains useful functions which allow the surface conditions 
# of the ice sheet to be inverted for the bed conditions beneath it
# This is the second version of this module. Updates include:
#      - Data files must be specified 
#      - Data is loaded with original coordinates
#      - Ability to select chosen version of transfer functions

## Import packages required for running code
import numpy as np
import itertools
from netCDF4 import Dataset
from scipy.interpolate import griddata 
from scipy.fft import fft2, ifft2
from file_processing import Write_to_nc
import time

def smooth_arr(B_ift):
    array_x = np.linspace(0, 2*np.pi, B_ift.shape[1])
    array_y = np.linspace(0, 2*np.pi, B_ift.shape[0])
    array_x2 = 0.5 - 0.5 * np.cos(array_x)
    array_y2 = 0.5 - 0.5 * np.cos(array_y)
    array_x2, array_y2 = np.meshgrid(array_x2, array_y2)
    array_ex2 = array_x2 * array_y2
    return array_ex2

def calcLM_KK(er, k, l, alpha_s, m, C, trans_funcs, rLM_KK= True, BtrueCfalse = True):
    """Function calculates the inverse transfer functions TBS, TBU, TBV"""
    """ if BtrueCFalse is True and TCS, TCU, TCV is BTrueCFalse is False"""
    
    if trans_funcs == 2008:
        from transferfuncs2008 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    elif trans_funcs == 2003:
        from transferfuncs2003 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    else:
        print('Error in transfer functions requested in calcLM_KK')
        
    err_S = er
    err_U = 1
    err_V = 1
    ES_sq = (1/err_S)** 2
    EU_sq = (1/err_U)** 2
    EV_sq = (1/err_V)** 2
    TSB = Tsb(k,l, alpha_s, m, C);
    TUB = Tub(k,l, alpha_s, m, C);
    TVB = Tvb(k,l, alpha_s, m, C);
    TSC = Tsc(k,l, alpha_s, m, C);
    TUC = Tuc(k,l, alpha_s, m, C);
    TVC = Tvc(k,l, alpha_s, m, C);
    L = (TSB * np.conj(TSB) * ES_sq) + (TUB * np.conj(TUB) * EU_sq) \
        + (TVB * np.conj(TVB) * EV_sq)
    M = (TSC * np.conj(TSC) * ES_sq) + (TUC * np.conj(TUC) * EU_sq) \
        + (TVC * np.conj(TVC) * EV_sq)
    K = (TSC * np.conj(TSB) * ES_sq) + (TUC * np.conj(TUB) * EU_sq)\
        + (TVC * np.conj(TVB) * EV_sq)
    LM_KK = (L * M) - (K * np.conj(K))
    LM_KK_inv = 1/LM_KK
    TBS = (LM_KK_inv) * (M * np.conj(TSB) - K * np.conj(TSC)) * ES_sq
    TBU = (LM_KK_inv) * (M * np.conj(TUB) - K * np.conj(TUC)) * EU_sq
    TBV = (LM_KK_inv) * (M * np.conj(TVB) - K * np.conj(TVC)) * EV_sq
    TCS = (LM_KK_inv) * (L * np.conj(TSC) - np.conj(K) * np.conj(TSB)) * ES_sq
    TCU = (LM_KK_inv) * (L * np.conj(TUC) - np.conj(K) * np.conj(TUB)) * EU_sq
    TCV = (LM_KK_inv) * (L * np.conj(TVC) - np.conj(K) * np.conj(TVB)) * EV_sq
    TBS[:,0] = 0 + 0j
    TBU[:,0] = 0 + 0j
    TBV[:,0] = 0 + 0j
    TCS[:,0] = 0 + 0j
    TCU[:,0] = 0 + 0j
    TCV[:,0] = 0 + 0j
    if rLM_KK == True:
        return LM_KK
    else:
        if BtrueCfalse == True:
            return TBS, TBU, TBV
        else:
            return TCS, TCU, TCV  

def filter(p, er, k, l, alpha_s, m, C, grid, trans_funcs):
    """To apply the filter to any grid of data"""
    LM_KK = calcLM_KK(er, k, l, alpha_s, m, C, trans_funcs, rLM_KK = True, BtrueCfalse=True)
    P = np.nanmax(LM_KK) * (C ** p)
    LM_KK_morethan_P = LM_KK >= P
    LM_KK_0 = LM_KK / P
    LM_KK_0[LM_KK_morethan_P] = 1 + 0j
    LM_KK_0[0,0] = 0 + 0j
    grid_filt_ft = LM_KK_0 * grid
    grid_filt = ifft2(grid_filt_ft).real
    return grid_filt_ft, grid_filt

def tapering_func(a, tapering):
    x_taper = int(a.shape[1] * tapering +1)
    x_taper_side = np.linspace(0,1,x_taper)
    x_taper_middle = np.ones(a.shape[1] - 2 * len(x_taper_side))
    x_taper2 = np.hstack([x_taper_side, x_taper_middle, np.flip(x_taper_side)])
    y_taper = int(a.shape[0] * tapering +1)
    y_taper_side = np.linspace(0,1,y_taper)
    y_taper_middle = np.ones(a.shape[0] - 2 * len(y_taper_side))
    y_taper2 = np.hstack([y_taper_side, y_taper_middle, np.flip(y_taper_side)])
    x, y = np.meshgrid(x_taper2, y_taper2)
    z = x* y
    return z

def smooth_data_load(bounds, filepath_itslive, filepath_rema, filepath_bedmach):
    ## Loading in the data for the inversion 
    # For the itslive data 
    fh = Dataset(filepath_itslive, 'r', format='NETCDF4');
    X = fh.variables['x'][:]
    Y = fh.variables['y'][:]
    xl = next(x for x, val in enumerate(X) if val >= bounds[0]) # X in ascending order
    xh = next(x for x, val in enumerate(X) if val >= bounds[1])
    yl = next(x for x, val in enumerate(Y) if val <= bounds[2])  # Y in descending order
    yh = next(x for x, val in enumerate(Y) if val <= bounds[3])  
    VX = fh.variables['VX'][yh:yl, xl:xh]
    VY = fh.variables['VY'][yh:yl, xl:xh]
    X_its = fh.variables['x'][xl:xh]
    Y_its = fh.variables['y'][yh:yl]
    X_its, Y_its = np.meshgrid(X_its, Y_its)
    fh.close()
    # For the rema data
    fh = Dataset(filepath_rema, 'r', format='NETCDF4');
    X = fh.variables['x'][:]
    Y = fh.variables['y'][:]
    xl = next(x for x, val in enumerate(X) if val >= bounds[0]) # X in ascending order
    xh = next(x for x, val in enumerate(X) if val >= bounds[1])
    yl = next(x for x, val in enumerate(Y) if val >= bounds[2])  # Y in descending order
    yh = next(x for x, val in enumerate(Y) if val >= bounds[3])  
    SURF = fh.variables['Band1'][yl:yh, xl:xh]
    X_rema = fh.variables['x'][xl:xh]
    Y_rema = fh.variables['y'][yl:yh]
    X_rema, Y_rema = np.meshgrid(X_rema, Y_rema)
    fh.close()
    # For the bedmachine data
    fh = Dataset(filepath_bedmach, 'r', format='NETCDF4');
    X = fh.variables['x'][:]
    Y = fh.variables['y'][:]
    xl = next(x for x, val in enumerate(X) if val >= bounds[0]) # X in ascending order
    xh = next(x for x, val in enumerate(X) if val >= bounds[1])
    yl = next(x for x, val in enumerate(Y) if val <= bounds[2])  # Y in descending order
    yh = next(x for x, val in enumerate(Y) if val <= bounds[3])  
    thick = fh.variables['thickness'][yh:yl, xl:xh]
    bed = fh.variables['bed'][yh:yl, xl:xh]
    errbed = fh.variables['errbed'][yh:yl, xl:xh]
    source = fh.variables['source'][yh:yl, xl:xh]
    X_bedmach = fh.variables['x'][xl:xh]
    Y_bedmach = fh.variables['y'][yh:yl]
    X_bedmach, Y_bedmach = np.meshgrid(X_bedmach, Y_bedmach)
    fh.close()
    return X_its, Y_its, VX, VY, X_rema, Y_rema, SURF, X_bedmach, Y_bedmach, thick, bed, errbed, source 

def bed_conditions_clean(centre_coord, tapering, p_b, p_c, erB, erC, m, C, square_size, trans_funcs, \
                         filepath_itslive, filepath_rema, filepath_bedmach, interp_grid_spacing, \
                             CutB, CutC, wavcutC, wavcutB):
    
    if trans_funcs == 2008:
        from transferfuncs2008 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    elif trans_funcs == 2003:
        from transferfuncs2003 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    else:
        print('Error in transfer functions requested in bed_conditions_clean')
    
    # From the central coordinate, create the boundaries of data to look at
    bounds = [centre_coord[0]-(square_size/2), centre_coord[0]+(square_size/2), 
          centre_coord[1]-(square_size/2), centre_coord[1] + (square_size/2)]
    big_bounds = [centre_coord[0]-(square_size), centre_coord[0]+(square_size), 
          centre_coord[1]-(square_size), centre_coord[1] + (square_size)]
    small_bounds = [centre_coord[0]-(square_size/4), centre_coord[0]+(square_size/4), 
          centre_coord[1]-(square_size/4), centre_coord[1] + (square_size/4)]
    # Load in the data in one region of interest
    X_its, Y_its, VX, VY, X_rema, Y_rema, SURF, X_bedmach, Y_bedmach, thick, bed, errbed, source = \
        smooth_data_load(bounds, filepath_itslive, filepath_rema, filepath_bedmach)
    
    ## Calculating the angle of slope and correcting the ice surface for this region of interest
    # To calculate the mean slope across the patch
    xs = np.ndarray.flatten(X_rema[::,::])
    ys = np.ndarray.flatten(Y_rema[::,::])
    zs = np.ndarray.flatten(SURF[::, ::])
    # Using matrix algebra to fit a plane to the 3D data
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    alpha_s = np.array((fit[0]**2 + fit[1] **2)/ (np.sqrt((fit[0]**2 + fit[1] **2))))
    angle = (np.arctan2(fit[1],fit[0]) *180/np.pi)
    # Calculate the mean ice thickness in this region
    h_bar = np.mean(thick) # Mean ice thickness
    
    ## The inversion works best when run on a data grid which is aligned to the direction of flow
    # Create a rotated grid to interpolate onto
    no_points = int(square_size/interp_grid_spacing) + 1 
    x_interp = np.linspace(0,interp_grid_spacing*(no_points-1), no_points)
    x_interp = x_interp - np.mean(x_interp)
    x_interp, y_interp = np.meshgrid(x_interp, x_interp)
    x_int = x_interp * np.cos(np.array(angle) * np.pi/180) - y_interp * np.sin(np.array(angle) * np.pi/180)
    x_int = x_int + centre_coord[0]
    y_int = y_interp * np.cos(np.array(angle) * np.pi/180) + x_interp * np.sin(np.array(angle) * np.pi/180)
    y_int = y_int + centre_coord[1]
    # Load the data to encompass this new grid 
    X_its, Y_its, VX, VY, X_rema, Y_rema, SURF, X_bedmach, Y_bedmach, thick, bed, errbed, source = \
        smooth_data_load(big_bounds, filepath_itslive, filepath_rema, filepath_bedmach)
    # Interpolate the velocity data
    X_its_flat = np.ndarray.flatten(X_its)
    Y_its_flat = np.ndarray.flatten(Y_its)
    VX_flat = np.ndarray.flatten(VX)
    VY_flat = np.ndarray.flatten(VY)
    VX_interp = griddata((X_its_flat, Y_its_flat), VX_flat, (x_int, y_int), method = 'cubic')
    VY_interp = griddata((X_its_flat, Y_its_flat), VY_flat, (x_int, y_int), method = 'cubic')
    # Recalculate VX and VY wrt the direction of flow
    ice_speed = np.sqrt(VX_interp ** 2 + VY_interp **2)
    theta = np.arctan2(VY_interp, VX_interp)
    VX_int =  ice_speed * np.cos(theta-(np.array(angle) * np.pi/180))
    VY_int =  ice_speed * np.sin(theta-(np.array(angle) * np.pi/180))
    # Calculate the mean ice speed in this grid
    u_bar = np.mean((ice_speed)) # Mean ice speed    
    # Interpolate the surface data
    X_rema_flat = np.ndarray.flatten(X_rema)
    Y_rema_flat = np.ndarray.flatten(Y_rema)
    SURF_flat = np.ndarray.flatten(SURF)
    SURF_int = griddata((X_rema_flat, Y_rema_flat),SURF_flat,(x_int, y_int), method = 'cubic')
    # Remove the surface slope from the surface data
    SLOPE_int = x_int * np.array(fit[0]) + y_int * np.array(fit[1]) + np.array(fit[2])
    SURF_int_corr = SURF_int - SLOPE_int

    ## Prepare the data for inversion
    # Remove the mean data values
    S_corr = SURF_int_corr - np.mean(SURF_int_corr)
    U_corr = VX_int - np.mean(VX_int)
    V_corr = VY_int - np.mean(VY_int)
    # Create trapeziod tapering/smoothing function
    z = tapering_func(x_int, tapering)
    # Non dimensionalise and smooth the edges
    S_4inv = S_corr / h_bar * z
    S_4inv = S_4inv - np.mean(S_4inv)
    U_4inv = U_corr / u_bar * z
    U_4inv = U_4inv - np.mean(U_4inv)
    V_4inv = V_corr / u_bar * z
    V_4inv = V_4inv - np.mean(V_4inv)
    # Fourier transform
    S_ft = fft2(S_4inv)
    U_ft = fft2(U_4inv)
    V_ft = fft2(V_4inv)
    # Create the arrays of k and l (directional wavenumbers) for the tranfer functions 
    ar1 = np.fft.fftfreq(SURF_int.shape[1], interp_grid_spacing/h_bar)
    ar2 = np.fft.fftfreq(SURF_int.shape[0], interp_grid_spacing/h_bar)
    k,l = np.meshgrid(ar1,ar2)
    j = np.sqrt(k**2 + l**2) # Non-directional wavenumber
    theta = np.arctan2(k,l) # Angle to flow/grid
    
    # Run the inversion
    # Smoothing the wavelengths of the input data  to remove noise (p_b and p_c)
    # Weighting it appropriately for least squares inversion (erB and erC)
    S_filt_ft_b, S_filta_b = filter(p_b, erB, k, l, alpha_s, m, C, S_ft, trans_funcs)
    U_filt_ft_b, U_filta_b = filter(p_b, erB, k, l, alpha_s, m, C, U_ft, trans_funcs)
    V_filt_ft_b, V_filta_b = filter(p_b, erB, k, l, alpha_s, m, C, V_ft, trans_funcs)
    S_filt_ft_c, S_filta_c = filter(p_c, erC, k, l, alpha_s, m, C, S_ft, trans_funcs)
    U_filt_ft_c, U_filta_c = filter(p_c, erC, k, l, alpha_s, m, C, U_ft, trans_funcs)
    V_filt_ft_c, V_filta_c = filter(p_c, erC, k, l, alpha_s, m, C, V_ft, trans_funcs)
    # Calculating the inverse transfer functions
    TBS, TBU, TBV = calcLM_KK(erB, k, l, alpha_s, m, C, trans_funcs, rLM_KK = False, BtrueCfalse = True)
    TCS, TCU, TCV = calcLM_KK(erC, k, l, alpha_s, m, C, trans_funcs, rLM_KK = False, BtrueCfalse = False)
    # Calculating the bed conditions, inverse fourier transform and rescaling
    B_ft2 = TBV * V_filt_ft_b + TBU * U_filt_ft_b + TBS * S_filt_ft_b
    C_ft2 = TCV * V_filt_ft_c + TCU * U_filt_ft_c + TCS * S_filt_ft_c
    # Remove any np.nan values that might cause errors in Fourier Transform
    B_ft2[np.isnan(B_ft2) ==True ] = 0
    C_ft2[np.isnan(C_ft2) ==True ] = 0
    # Remove any features aligned with the direction of flow (CutB, CutC)
    # Remove some problematic small wavelength features (wavcutB, wavcutC, ratios of h_bar) 
    maskCutB1 = np.abs(theta) < (CutB * np.pi/180) 
    maskCutB2 = np.abs(theta) > ((180 -CutB) * np.pi/180) 
    maskCutB = maskCutB1 + maskCutB2
    B_ft2[maskCutB] = 0
    maskwavcutB = ((1/j)*h_bar) < wavcutB * h_bar
    B_ft2[maskwavcutB] = 0
    maskCutC = (np.abs(theta) > ((90-CutC)*np.pi/180)) & (np.abs(theta) < ((90+CutC)*np.pi/180))
    C_ft2[maskCutC] = 0
    maskwavcutC = ((1/j)*h_bar) <  wavcutC * h_bar
    C_ft2[maskwavcutC] = 0
    # Inverse Fourier transform
    B_ift2 = (ifft2(B_ft2) * h_bar) + SLOPE_int - (h_bar)
    C_ift2 = ifft2(C_ft2) * C + C
    
    # For a regional data product, we use the standard polar stereographic grid
    # Extract the relevant part of the REMA grid
    xl = next(x for x, val in enumerate(X_rema[0]) if val >= small_bounds[0]) # X in ascending order
    xh = next(x for x, val in enumerate(X_rema[0]) if val >= small_bounds[1])
    yl = next(x for x, val in enumerate(Y_rema[:,0]) if val >= small_bounds[2])  # Y in descending order
    yh = next(x for x, val in enumerate(Y_rema[:,0]) if val >= small_bounds[3])  
    X_rema_small = X_rema[yl:yh, xl:xh]
    Y_rema_small = Y_rema[yl:yh, xl:xh]
    rema_small = SURF[yl:yh, xl:xh]
    # Interpolate the bed data
    x_int_flat = np.ndarray.flatten(x_int)
    y_int_flat = np.ndarray.flatten(y_int)
    B_ift2_small = np.ndarray.flatten(B_ift2.real)
    C_ift2_small = np.ndarray.flatten(C_ift2.real)
    B_ift_interp = griddata((x_int_flat, y_int_flat), B_ift2_small, (X_rema_small, Y_rema_small), method = 'cubic')
    C_ift_interp = griddata((x_int_flat, y_int_flat), C_ift2_small, (X_rema_small, Y_rema_small), method = 'cubic')
    # In case it is useful, interpolate the bedmach data onto this grid too 
    # (can remove this section from the code later if needed)
    X_bedmach_flat = np.ndarray.flatten(X_bedmach)
    Y_bedmach_flat = np.ndarray.flatten(Y_bedmach)
    bed_flat = np.ndarray.flatten(bed)
    errbed_flat = np.ndarray.flatten(errbed)
    bed_interp = griddata((X_bedmach_flat, Y_bedmach_flat), bed_flat, (X_rema_small, Y_rema_small), method = 'cubic')
    errbed_interp = griddata((X_bedmach_flat, Y_bedmach_flat), errbed_flat, (X_rema_small, Y_rema_small), method = 'cubic')

    return B_ift_interp, C_ift_interp, X_rema_small, Y_rema_small

def terminal_inversion_smooth(m, C, p_b, p_c, erB, erC, n, adj, square_size, tapering, centre_include, \
                              centre_coord, trans_funcs, \
                                  filepath_itslive, filepath_rema, filepath_bedmach, interp_grid_spacing, \
                                  CutB, CutC, wavcutC, wavcutB, filename):
    
    if trans_funcs == 2008:
        from transferfuncs2008 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    elif trans_funcs == 2003:
        from transferfuncs2003 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    else:
        print('Error in transfer functions requested')
    
    outer_X_min = centre_coord[0] - ((square_size*adj[0])/(n*4))
    outer_X_max = centre_coord[0] + ((square_size*adj[0])/(n*4))
    outer_Y_min = centre_coord[1] - ((square_size*adj[0])/(n*4))
    outer_Y_max = centre_coord[1] + ((square_size*adj[0])/(n*4))
    outer_X_min = centre_coord[0] - ((square_size*3)/(n*4))
    outer_X_max = centre_coord[0] + ((square_size*3)/(n*4))
    outer_Y_min = centre_coord[1] - ((square_size*3)/(n*4))
    outer_Y_max = centre_coord[1] + ((square_size*3)/(n*4))
  
    centresx = np.arange(0,(adj[0]+n-1),1) * (square_size/2)/(centre_include *n)
    centresx = centresx - (centresx.max() - centresx.min())/2
    centresy = np.arange(0,(adj[1]+n-1),1) * (square_size/2)/(centre_include *n)
    centresy = centresy - (centresy.max() - centresy.min())/2
    centrex, centrey = np.meshgrid(centresx, centresy)
    centrex_adj = centre_coord[0]+ centrex
    centrey_adj = centre_coord[1] + centrey

    # These are the lines which take a long time
    X_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    Y_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    B_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    C_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    print('Starting Inversion now')
    #for i,j in itertools.product(range(adj[0]),range(adj[1])):
    for i,j in itertools.product(range(len(centrex_adj[0])), range(len(centrex_adj[1]))):
        B_adjs[i,j], C_adjs[i,j], X_adjs[i,j], Y_adjs[i,j] = \
        bed_conditions_clean([centrex_adj[i,j], centrey_adj[i,j]], tapering, p_b, p_c, erB, erC, \
                             m, C, square_size, trans_funcs, \
                                 filepath_itslive, filepath_rema, filepath_bedmach, interp_grid_spacing, \
                                 CutB, CutC, wavcutC, wavcutB)
        if j == len(centrex_adj[1])-1:
            print((i +1) * (j +1),'grids out of',len(centrex_adj[0]) * (j+1),'processed')
    print('Ending Inversion now')
    
    mins_X = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    maxs_X = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    mins_Y = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    maxs_Y = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        mins_X[i,j] = np.min(X_adjs[i,j])
        maxs_X[i,j] = np.max(X_adjs[i,j])
        mins_Y[i,j] = np.min(Y_adjs[i,j])
        maxs_Y[i,j] = np.max(Y_adjs[i,j])
    max_X = np.max(maxs_X)
    min_X = np.min(mins_X)
    max_Y = np.max(maxs_Y)
    min_Y = np.min(mins_Y)
    #max_Y = np.max((Y_adjs[0,0].max(), Y_adjs[Y_adjs.shape[0]-1, Y_adjs.shape[0]-1].max()))
    #min_Y = np.min((Y_adjs[0,0].min(), Y_adjs[Y_adjs.shape[0]-1, Y_adjs.shape[0]-1].min()))
    overall_X = np.arange(min_X, max_X+1, interp_grid_spacing)
    overall_Y = np.arange(min_Y, max_Y+1, interp_grid_spacing)
    #    overall_Y = np.arange(max_Y, min_Y-1, -interp_grid_spacing)
    # Required for the previous dataset where the y coordinates were upside down
    big_X, big_Y = np.meshgrid(overall_X, overall_Y)

    big_bed = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_w = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_slip = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_bed2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_w2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_slip2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)

    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        length = X_adjs[i,j].shape
        xl = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][0,0])) 
        xh = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][length[0]-1, length[1]-1]))
        yl = next(x for x, val in enumerate(overall_Y)\
                           if val >= (Y_adjs[i,j][0,0])) 
        yh = next(x for x, val in enumerate(overall_Y)\
                            if val >= (Y_adjs[i,j][length[0]-1, length[1]-1]))
        big_bed[i,j] = np.zeros(big_X.shape)
        big_w[i,j] = np.zeros(big_X.shape)
        big_bed[i,j][yl:yh+1, xl:xh+1] = B_adjs[i,j]
        big_w[i,j][yl:yh+1, xl:xh+1] = smooth_arr(B_adjs[i,j])
        big_slip[i,j] = np.zeros(big_X.shape)
        big_slip[i,j][yl:yh+1, xl:xh+1] = C_adjs[i,j] 
    big_bed_total = np.sum(big_bed * big_w)/(np.sum(big_w))
    big_slip_total = np.sum(big_slip * big_w)/(np.sum(big_w))
    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        length = X_adjs[i,j].shape
        xl = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][0,0])) 
        xh = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][length[0]-1, length[1]-1]))
        yl = next(x for x, val in enumerate(overall_Y)\
                           if val >= (Y_adjs[i,j][0,0])) 
        yh = next(x for x, val in enumerate(overall_Y)\
                            if val >= (Y_adjs[i,j][length[0]-1, length[1]-1]))
        big_bed2[i,j] = np.zeros((big_X.shape)) #* np.nan
        big_w2[i,j] = np.zeros((big_X.shape)) #* np.nan
      #  big_bed2[i,j][yl:yh+1, xl:xh+1] = B_adjs[i,j]
        big_w2[i,j][yl:yh+1, xl:xh+1] = smooth_arr(B_adjs[i,j])
        big_slip2[i,j] = np.zeros((big_X.shape)) #* np.nan
      #  big_slip2[i,j][yl:yh+1, xl:xh+1] = C_adjs[i,j]
        big_bed2[i,j] = np.abs((big_bed[i,j] - big_bed_total)**2)
    #    big_bed2[i,j][big_bed[i,j] == 0] = 0
        big_slip2[i,j] =np.abs((big_slip[i,j] - big_slip_total)**2)
    #    big_slip2[i,j][big_bed[i,j] == 0] = 0

    big_bed2_total = np.sum(big_bed2 * big_w2)/(np.sum(big_w2))
    big_bed2_total = np.sqrt(big_bed2_total)
    big_slip2_total = np.sum(big_slip2 * big_w2)/(np.sum(big_w2))
    big_slip2_total = np.sqrt(big_slip2_total)
    
    # Calculate which bit of the grid to include
    width_x = square_size/2 * adj[0] / (n * centre_include)
    width_y = square_size/2 * adj[1] / (n * centre_include)
    outer_X_min = centre_coord[0] - width_x/2
    outer_X_max = centre_coord[0] + width_x/2
    outer_Y_min = centre_coord[1] - width_y/2
    outer_Y_max = centre_coord[1] + width_y/2
    xl = next(x for x, val in enumerate(overall_X)\
                           if val >= outer_X_min) 
    xh = next(x for x, val in enumerate(overall_X)\
                           if val >= outer_X_max)
    yl = next(x for x, val in enumerate(overall_Y)\
                           if val >= outer_Y_min) 
    yh = next(x for x, val in enumerate(overall_Y)\
                            if val >= outer_Y_max)
    overall_X2 = big_X[yl:yh, xl:xh]
    overall_Y2 = big_Y[yl:yh, xl:xh]
    overall_bed = big_bed_total[yl:yh,xl:xh]
    overall_slip = big_slip_total[yl:yh,xl:xh]
    overall_b_std = big_bed2_total[yl:yh,xl:xh]
    overall_c_std = big_slip2_total[yl:yh,xl:xh]

    #Save results to file 
    Write_to_nc(overall_X2, overall_Y2, overall_bed, overall_bed, overall_b_std, \
                overall_slip, overall_c_std, filename)
    print('Results saved to file')

def smooth_data_load_rema_its(bounds, filepath_rema_its):
    ## Loading in the data for the inversion 
    # For the itslive data 
    fh = Dataset(filepath_rema_its, 'r', format='NETCDF4');
    X = fh.variables['x'][:]
    Y = fh.variables['y'][:]
    xl = next(x for x, val in enumerate(X) if val >= bounds[0]) # X in ascending order
    xh = next(x for x, val in enumerate(X) if val >= bounds[1])
    yl = next(x for x, val in enumerate(Y) if val <= bounds[2])  # Y in descending order
    yh = next(x for x, val in enumerate(Y) if val <= bounds[3])  
    SURF = fh.variables['Band1'][yh:yl, xl:xh]
    X_rema = fh.variables['x'][xl:xh]
    Y_rema = fh.variables['y'][yh:yl]
    X_rema, Y_rema = np.meshgrid(X_rema, Y_rema)
    fh.close()
    return X_rema, Y_rema, SURF

def bed_conditions_frot3(centre_coord, tapering, n, p_b, p_c, erB, erC, m, C, square_size, trans_funcs, \
                         filepath_itslive, filepath_rema, filepath_rema_its, filepath_bedmach, interp_grid_spacing, \
                             CutB, CutC, wavcutC, wavcutB):
    
    if trans_funcs == 2008:
        from transferfuncs2008 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    elif trans_funcs == 2003:
        from transferfuncs2003 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    else:
        print('Error in transfer functions requested in bed_conditions_clean')
        
    # From the central coordinate, create the boundaries of data to look at
    # But since there's no interpolation this is the only grid we need 
    bounds = [centre_coord[0]-(square_size/2), centre_coord[0]+(square_size/2), 
          centre_coord[1]-(square_size/2), centre_coord[1] + (square_size/2)]
    big_bounds = [centre_coord[0]-(square_size/1.5), centre_coord[0]+(square_size/1.5), 
              centre_coord[1]-(square_size/1.5), centre_coord[1] + (square_size/1.5)]
    # Load in the data in one region of interest
    X_its, Y_its, VX, VY, _, _, _, X_bedmach, Y_bedmach, thick, bed, errbed, source = \
        smooth_data_load(bounds, filepath_itslive, filepath_rema, filepath_bedmach)
    X_rema, Y_rema, SURF =  \
        smooth_data_load_rema_its(bounds, filepath_rema_its)
#    Y_rema = np.flip(Y_rema, axis = 0)
 #   SURF = np.flip(SURF, axis = 0)
      
    ## Calculating the angle of slope and correcting the ice surface for this region of interest
    # To calculate the mean slope across the patch
    xs = np.ndarray.flatten(X_rema[::,::])
    ys = np.ndarray.flatten(Y_rema[::,::])
    zs = np.ndarray.flatten(SURF[::, ::])
    # Using matrix algebra to fit a plane to the 3D data
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    alpha_s = np.array((fit[0]**2 + fit[1] **2)/ (np.sqrt((fit[0]**2 + fit[1] **2))))
    angle = -(np.arctan2(fit[1],fit[0]) *180/np.pi)
    # Calculate the mean ice thickness in this region
#     h_bar = np.mean(thick) # Mean ice thickness
#     print('whole h_bar')
    low_frac = (2*n -1)/(4*n)
    high_frac = (2*n +1)/(4*n)
    x_low = np.int(np.round(low_frac * X_bedmach.shape[1]))
    x_high = np.int(np.round(high_frac * X_bedmach.shape[1]))
    y_low = np.int(np.round(low_frac * Y_bedmach.shape[1]))
    y_high = np.int(np.round(high_frac * Y_bedmach.shape[1]))
    h_bar = np.mean(thick[y_low:y_high, x_low:x_high])
    
    # Remove the surface slope from the surface data
    SLOPE_its = X_rema * np.array(fit[0]) + Y_rema * np.array(fit[1]) + np.array(fit[2])
    SURF_interp = SURF - SLOPE_its
    
    # Recalculate the velocity fields so that VX is the speed in the direction of flow 
    ice_speed = np.sqrt(VX**2 + VY**2)
    theta = np.arctan2(VY, VX)
    ice_direct = np.mean(np.arctan2(VY, VX)) *180/np.pi
    VX_rot =  ice_speed * np.cos(theta-(np.array(ice_direct) * np.pi/180))
    VY_rot =  ice_speed * np.sin(theta-(np.array(ice_direct) * np.pi/180))
    
    if C == True:
        C = -6.208 * (np.log(h_bar*alpha_s/np.mean(ice_speed))) + 47.644
    else:
        None
    
#     # Interpolate the surface data onto the velocity grid 
#     X_rema_flat = np.ndarray.flatten(X_rema)
#     Y_rema_flat = np.ndarray.flatten(Y_rema)
#     SURF_flat = np.ndarray.flatten(SURF_int_corr)
#     SURF_interp = griddata((X_rema_flat, Y_rema_flat), SURF_flat, (X_its, Y_its), method = 'cubic')
#     # Calculate the surface on the new grid to interpolate back later? 
#     SLOPE_its = X_its * np.array(fit[0]) + Y_its * np.array(fit[1]) + np.array(fit[2])

    #h_bar = np.mean(thick)
    Ice_speed = np.sqrt(VX**2 + VY**2)
    mean_ice_vel = np.mean((Ice_speed))
    u_bar = mean_ice_vel
    spacing = np.abs(X_its[0,0]-X_its[0,1])
    edge_smoothing = ((np.nanmax(Y_its) - np.nanmin(Y_its)) * tapering)
    edge_w = int(edge_smoothing/spacing)
    tap1 = np.linspace(0,1,edge_w)
    tap2x = np.ones(X_its.shape[1]- 2* edge_w)
    tap2y = np.ones(X_its.shape[0]- 2* edge_w)
    tap3 = np.flip(tap1)
    tapx = np.hstack([tap1,tap2x,tap3])
    tapy = np.hstack([tap1,tap2y,tap3])
    X1, Y1 = np.meshgrid(tapx,tapy)
    z = X1*Y1

    S_corr = SURF_interp
    S_corr = S_corr - np.mean(S_corr)
    U_corr = VX_rot - np.mean(VX_rot)
    V_corr = VY_rot - np.mean(VY_rot)

    # Non dimensionalise and smooth the edges
    S_4inv = S_corr / h_bar * z
    S_4inv = S_4inv - np.mean(S_4inv)
    U_4inv = U_corr / u_bar * z
    U_4inv = U_4inv - np.mean(U_4inv)
    V_4inv = V_corr / u_bar * z
    V_4inv = V_4inv - np.mean(V_4inv)
    # Fourier transform
    S_ft = fft2(S_4inv)
    U_ft = fft2(U_4inv)
    V_ft = fft2(V_4inv)
    # Create the arrays of k and l for the tranfer functions 
    ar1 = np.fft.fftfreq(X_its.shape[1], spacing/h_bar)
    ar2 = np.fft.fftfreq(X_its.shape[0], spacing/h_bar)
    k,l = np.meshgrid(ar1,ar2)
    angle2 = angle
    #angle2 = ice_direct
    j = np.sqrt( k **2 + l **2)
    theta = np.arctan2(l,k) * (180/np.pi)
    # Recalculate the transfer functions in the right direction
    wave_angle_to_slope = theta - np.array(angle2) 
    k_prime = j * np.cos(wave_angle_to_slope * np.pi/180)
    l_prime = j * np.sin(wave_angle_to_slope * np.pi/180)
    theta_prime = np.arctan2(l_prime,k_prime) * (180/np.pi)

    # Smoothing the input data and weighting it appropriately
    S_filt_ft_b, S_filta_b = filter(p_b, erB, k_prime, l_prime, alpha_s, m, C, S_ft, trans_funcs)
    U_filt_ft_b, U_filta_b = filter(p_b, erB, k_prime, l_prime, alpha_s, m, C, U_ft, trans_funcs)
    V_filt_ft_b, V_filta_b = filter(p_b, erB, k_prime, l_prime, alpha_s, m, C, V_ft, trans_funcs)
    S_filt_ft_c, S_filta_c = filter(p_c, erC, k_prime, l_prime, alpha_s, m, C, S_ft, trans_funcs)
    U_filt_ft_c, U_filta_c = filter(p_c, erC, k_prime, l_prime, alpha_s, m, C, U_ft, trans_funcs)
    V_filt_ft_c, V_filta_c = filter(p_c, erC, k_prime, l_prime, alpha_s, m, C, V_ft, trans_funcs)
    # Calculating the inverse transfer functions
    TBS, TBU, TBV = calcLM_KK(erB, k_prime, l_prime, alpha_s, m, C, trans_funcs, rLM_KK = False, BtrueCfalse = True)
    TCS, TCU, TCV = calcLM_KK(erC, k_prime, l_prime, alpha_s, m, C, trans_funcs, rLM_KK = False, BtrueCfalse = False)
    # Calculating the bed conditions, inverse fourier transform and rescaling
    B_ft = TBV * V_filt_ft_b + TBU * U_filt_ft_b + TBS * S_filt_ft_b
    C_ft = TCV * V_filt_ft_c + TCU * U_filt_ft_c + TCS * S_filt_ft_c
    B_ft[np.isnan(B_ft) ==True ] = 0
    C_ft[np.isnan(C_ft) ==True ] = 0
    
#     theta = np.arctan2(l_prime,k_prime)
#     #CutB = 5
#     CutB = CutB * np.pi/180
#     maskCutB = (np.abs(theta) < (CutB + np.pi/2)) & (np.abs(theta) > (np.pi/2 - CutB))
#     maskwavcutB = ((1/j)*h_bar) < wavcutB * h_bar
#     B_ft2[maskwavcutB] = 0
#     #CutC = 5
    CutC = CutC * np.pi/180
    maskCutC = (np.abs(theta) < (CutC + np.pi/2)) & (np.abs(theta) > (np.pi/2 - CutC))
    maskCutC2 = (np.abs(theta) < CutC) + (np.abs(theta) > (np.pi - CutC))
    C_ft[maskCutC + maskCutC2] = 0
    maskwavcutC = ((1/j)*h_bar) <  wavcutC * h_bar
    C_ft[maskwavcutC] = 0
    
    theta = np.arctan2(l_prime,k_prime)
    theta2 = theta * (-np.cos((theta/CutB) * np.pi) + 1)/2
    CutB = CutB * np.pi/180
    mask = (np.abs(theta) < (CutB + np.pi/2)) & (np.abs(theta) > (np.pi/2 - CutB))
    maskand = (np.abs(theta) < CutB) + (np.abs(theta) > (np.pi - CutB))
    B_ft[mask] = B_ft[mask] * theta2[mask]
    
#     # Remove any features aligned with the direction of flow (CutB, CutC)
#     # Remove some problematic small wavelength features (wavcutB, wavcutC, ratios of h_bar) 
#     maskCutB1 = np.abs(theta) < (CutB * np.pi/180) 
#     maskCutB2 = np.abs(theta) > ((180 -CutB) * np.pi/180) 
#     maskCutB = maskCutB1 + maskCutB2
#     B_ft[maskCutB] = 0
#     maskwavcutB = ((1/j)*h_bar) < wavcutB * h_bar
#     B_ft[maskwavcutB] = 0
#     maskCutC = (np.abs(theta) > ((90-CutC)*np.pi/180)) & (np.abs(theta) < ((90+CutC)*np.pi/180))
#     C_ft[maskCutC] = 0
#     maskwavcutC = ((1/j)*h_bar) <  wavcutC * h_bar
#     C_ft[maskwavcutC] = 0

    B_ift = (ifft2(B_ft) * h_bar) + SLOPE_its - (h_bar)
    C_ift = ifft2(C_ft) * C + C

    width_x = square_size/2 #* adj[0] / (n * centre_include)
    width_y = square_size/2 #* adj[1] / (n * centre_include)
    outer_X_min = centre_coord[0] - width_x/2
    outer_X_max = centre_coord[0] + width_x/2
    outer_Y_min = centre_coord[1] - width_y/2
    outer_Y_max = centre_coord[1] + width_y/2
    xl = next(x for x, val in enumerate(X_its[0,:])\
                           if val >= outer_X_min) 
    xh = next(x for x, val in enumerate(X_its[0,:])\
                           if val >= outer_X_max)
    yl = next(x for x, val in enumerate(Y_its[:,0])\
                           if val <= outer_Y_max) 
    yh = next(x for x, val in enumerate(Y_its[:,0])\
                           if val <= outer_Y_min)
    B_ret = B_ift.real[yl:yh,xl:xh]
    C_ret = C_ift.real[yl:yh,xl:xh]
    X2_ret = X_its[yl:yh,xl:xh]
    Y2_ret = Y_its[yl:yh,xl:xh]
    return B_ret, C_ret, X2_ret, Y2_ret

def terminal_inversion_smooth_frot(m, C, p_b, p_c, erB, erC, n, adj, square_size, tapering, centre_include, \
                              centre_coord, trans_funcs, \
                                  filepath_itslive, filepath_rema, filepath_rema_its, filepath_bedmach, interp_grid_spacing, \
                                  CutB, CutC, wavcutC, wavcutB, filename):
    
    if trans_funcs == 2008:
        from transferfuncs2008 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    elif trans_funcs == 2003:
        from transferfuncs2003 import Tsb, Tub, Tvb, Tsc, Tuc, Tvc
    else:
        print('Error in transfer functions requested')
    
    outer_X_min = centre_coord[0] - ((square_size*adj[0])/(n*4))
    outer_X_max = centre_coord[0] + ((square_size*adj[0])/(n*4))
    outer_Y_min = centre_coord[1] - ((square_size*adj[0])/(n*4))
    outer_Y_max = centre_coord[1] + ((square_size*adj[0])/(n*4))
    outer_X_min = centre_coord[0] - ((square_size*3)/(n*4))
    outer_X_max = centre_coord[0] + ((square_size*3)/(n*4))
    outer_Y_min = centre_coord[1] - ((square_size*3)/(n*4))
    outer_Y_max = centre_coord[1] + ((square_size*3)/(n*4))
  
    centresx = np.arange(0,(adj[0]+n-1),1) * (square_size/2)/(centre_include *n)
    centresx = centresx - (centresx.max() - centresx.min())/2
    centresy = np.arange(0,(adj[1]+n-1),1) * (square_size/2)/(centre_include *n)
    centresy = centresy - (centresy.max() - centresy.min())/2
    centrex, centrey = np.meshgrid(centresx, centresy)
    centrex_adj = centre_coord[0]+ centrex
    centrey_adj = centre_coord[1] + centrey

    # These are the lines which take a long time
    # Record start time
    start_time = time.time()
    X_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    Y_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    B_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    C_adjs = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    print('Starting Inversion now')
    #for i,j in itertools.product(range(adj[0]),range(adj[1])):
    for i,j in itertools.product(range(len(centrex_adj[0])), range(len(centrex_adj[1]))):
        B_adjs[i,j], C_adjs[i,j], X_adjs[i,j], Y_adjs[i,j] = \
        bed_conditions_frot3([centrex_adj[i,j], centrey_adj[i,j]], tapering, n, p_b, p_c, erB, erC, \
                             m, C, square_size, trans_funcs, \
                                 filepath_itslive, filepath_rema, filepath_rema_its, filepath_bedmach, interp_grid_spacing, \
                                 CutB, CutC, wavcutC, wavcutB)
        if j == len(centrex_adj[1])-1:
            print((i +1) * (j +1),'grids out of',len(centrex_adj[0]) * (j+1),'processed', end='\r')
    #print('Ending Inversion now')
    # Record end time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Ending Inversion now: {elapsed_time:.4f} seconds")
    
    mins_X = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    maxs_X = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    mins_Y = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    maxs_Y = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]))
    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        mins_X[i,j] = np.min(X_adjs[i,j])
        maxs_X[i,j] = np.max(X_adjs[i,j])
        mins_Y[i,j] = np.min(Y_adjs[i,j])
        maxs_Y[i,j] = np.max(Y_adjs[i,j])
    max_X = np.max(maxs_X)
    min_X = np.min(mins_X)
    max_Y = np.max(maxs_Y)
    min_Y = np.min(mins_Y)
    #max_Y = np.max((Y_adjs[0,0].max(), Y_adjs[Y_adjs.shape[0]-1, Y_adjs.shape[0]-1].max()))
    #min_Y = np.min((Y_adjs[0,0].min(), Y_adjs[Y_adjs.shape[0]-1, Y_adjs.shape[0]-1].min()))
    overall_X = np.arange(min_X, max_X+1, interp_grid_spacing)
    overall_Y = np.arange(min_Y, max_Y+1, interp_grid_spacing)
    #    overall_Y = np.arange(max_Y, min_Y-1, -interp_grid_spacing)
    # Required for the previous dataset where the y coordinates were upside down
    big_X, big_Y = np.meshgrid(overall_X, overall_Y)

    big_bed = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_w = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_slip = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_bed2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_w2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)
    big_slip2 = np.zeros((centrex_adj.shape[0],centrex_adj.shape[1]), dtype=np.ndarray)

    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        length = X_adjs[i,j].shape
        xl = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][0,0])) 
        xh = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][length[0]-1, length[1]-1]))
        yh = next(x for x, val in enumerate(overall_Y)\
                           if val >= (Y_adjs[i,j][0,0])) 
        yl = next(x for x, val in enumerate(overall_Y)\
                            if val >= (Y_adjs[i,j][length[0]-1, length[1]-1]))
        big_bed[i,j] = np.zeros(big_X.shape)
        big_w[i,j] = np.zeros(big_X.shape)
        big_bed[i,j][yl:yh+1, xl:xh+1] = np.flip(B_adjs[i,j], axis = 0)
        big_w[i,j][yl:yh+1, xl:xh+1] = smooth_arr(B_adjs[i,j])
        big_slip[i,j] = np.zeros(big_X.shape)
        big_slip[i,j][yl:yh+1, xl:xh+1] = np.flip(C_adjs[i,j], axis = 0) 
    big_bed_total = np.sum(big_bed * big_w)/(np.sum(big_w))
    big_slip_total = np.sum(big_slip * big_w)/(np.sum(big_w))
    for i,j in itertools.product(range(centrex_adj.shape[0]), range(centrey_adj.shape[1])):
        length = X_adjs[i,j].shape
        xl = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][0,0])) 
        xh = next(x for x, val in enumerate(overall_X)\
                           if val >= (X_adjs[i,j][length[0]-1, length[1]-1]))
        yh = next(x for x, val in enumerate(overall_Y)\
                           if val >= (Y_adjs[i,j][0,0])) 
        yl = next(x for x, val in enumerate(overall_Y)\
                            if val >= (Y_adjs[i,j][length[0]-1, length[1]-1]))
        big_bed2[i,j] = np.zeros((big_X.shape)) #* np.nan
        big_w2[i,j] = np.zeros((big_X.shape)) #* np.nan
      #  big_bed2[i,j][yl:yh+1, xl:xh+1] = B_adjs[i,j]
        big_w2[i,j][yl:yh+1, xl:xh+1] = smooth_arr(B_adjs[i,j])
        big_slip2[i,j] = np.zeros((big_X.shape)) #* np.nan
      #  big_slip2[i,j][yl:yh+1, xl:xh+1] = C_adjs[i,j]
        big_bed2[i,j] = np.abs((big_bed[i,j] - big_bed_total)**2)
    #    big_bed2[i,j][big_bed[i,j] == 0] = 0
        big_slip2[i,j] =np.abs((big_slip[i,j] - big_slip_total)**2)
    #    big_slip2[i,j][big_bed[i,j] == 0] = 0

    big_bed2_total = np.sum(big_bed2 * big_w2)/(np.sum(big_w2))
    big_bed2_total = np.sqrt(big_bed2_total)
    big_slip2_total = np.sum(big_slip2 * big_w2)/(np.sum(big_w2))
    big_slip2_total = np.sqrt(big_slip2_total)
    
    # Calculate which bit of the grid to include
    width_x = square_size/2 * adj[0] / (n * centre_include)
    width_y = square_size/2 * adj[1] / (n * centre_include)
    outer_X_min = centre_coord[0] - width_x/2
    outer_X_max = centre_coord[0] + width_x/2
    outer_Y_min = centre_coord[1] - width_y/2
    outer_Y_max = centre_coord[1] + width_y/2
    xl = next(x for x, val in enumerate(overall_X)\
                           if val >= outer_X_min) 
    xh = next(x for x, val in enumerate(overall_X)\
                           if val >= outer_X_max)
    yl = next(x for x, val in enumerate(overall_Y)\
                           if val >= outer_Y_min) 
    yh = next(x for x, val in enumerate(overall_Y)\
                            if val >= outer_Y_max)
    overall_X2 = big_X[yl:yh, xl:xh]
    overall_Y2 = big_Y[yl:yh, xl:xh]
    overall_bed = big_bed_total[yl:yh,xl:xh]
    overall_slip = big_slip_total[yl:yh,xl:xh]
    overall_b_std = big_bed2_total[yl:yh,xl:xh]
    overall_c_std = big_slip2_total[yl:yh,xl:xh]

    #Save results to file 
    Write_to_nc(overall_X2, overall_Y2, overall_bed, overall_bed, overall_b_std, \
                overall_slip, overall_c_std, filename)
    print('Results saved to file')