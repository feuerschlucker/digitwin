import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def calc_metric(data, sample_ids, metric, x_arr):
    # Get the sorted indices
    sample_ids = np.array(sample_ids)
    sorted_indices = np.argsort(sample_ids)
    sample_ids = sample_ids[sorted_indices]
    data = data[sorted_indices]
    data_of_metric = [dat[metric] for dat in data]
    # Calculate linear regression
    y = data_of_metric
    x = x_arr
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # calc Mean Absolute Percentage Error (MAPE)
    y_pred = slope * x + intercept
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    # Calculate Coefficient of Variation (CV)
    std_dev = np.std(y - y_pred)
    mean_y = np.mean(y)
    cv = (std_dev / mean_y) * 100
    return x, y, slope, intercept, r_value, mape, cv, sample_ids


def get_normalized_color(min_col_val, max_col_val, col_val):
    cmap = plt.cm.get_cmap('jet')
    color_val = (col_val - min_col_val) / (max_col_val - min_col_val)
    color = cmap(color_val)
    return color, color_val


def get_data_matrix_dictionary(data_dict, x_arr):
    # get curve parameters
    curve_params = list(data_dict['data'][0, 0, 0].keys())

    images = {}
    for param in tqdm(curve_params):
        img = np.zeros((len(data_dict['amplitudes']), len(data_dict['frequencys']), 3))
        meta_data = {}
        meta_data['x'] = np.zeros(
            (len(data_dict['amplitudes']), len(data_dict['frequencys']), len(data_dict['sample_ids'])))
        meta_data['y'] = np.zeros(
            (len(data_dict['amplitudes']), len(data_dict['frequencys']), len(data_dict['sample_ids'])))
        meta_data['k'] = np.zeros((len(data_dict['amplitudes']), len(data_dict['frequencys']), 1))
        meta_data['d'] = np.zeros((len(data_dict['amplitudes']), len(data_dict['frequencys']), 1))
        for ii, amplitude in enumerate(data_dict['amplitudes']):
            for jj, frequency in enumerate(data_dict['frequencys']):
                x, y, k, d, rv, mape, cv, sample_ids = calc_metric(data_dict['data'][ii, jj, :], data_dict['sample_ids'], param,
                                                       x_arr)
                img[ii, jj, 0] = rv * rv  # R² Value
                img[ii, jj, 1] = -mape  # MAPE Value (negative)
                img[ii, jj, 2] = -cv  # Coefficient of Variation (negative)
                meta_data['x'][ii, jj, :] = x
                meta_data['y'][ii, jj, :] = y
                meta_data['sample_ids'] = sample_ids
                meta_data['k'] = k
                meta_data['d'] = d
        images[param] = {}
        images[param]['R²'] = img[:, :, 0]
        images[param]['MAPE'] = img[:, :, 1]
        images[param]['CV'] = img[:, :, 2]
        images[param]['meta_data'] = meta_data
    return images, curve_params


def get_boundary_dict(curve_params, images, metrics):
    # get plot boundaries
    bound_dict = {}
    for param in curve_params:
        bound_dict[param] = {}
        for metric in metrics:
            img_array = images[param][metric]
            height, width = np.shape(img_array)
            vals = []
            for ii in range(height):
                for jj in range(width):
                    vals.append(img_array[ii, jj])
            min_val = np.min(np.array(vals))
            max_val = np.max(np.array(vals))
            bound_dict[param][metric] = (min_val, max_val)
    return bound_dict


def combine_boundary_dicts(dict_list, curve_params, metrics):
    boundary_dict = {}
    for param in curve_params:
        boundary_dict[param] = {}
        for metric in metrics:
            min_ = float('inf')
            max_ = float('-inf')
            for dict_ in dict_list:
                min_val = dict_[param][metric][0]
                max_val = dict_[param][metric][1]
                if min_val < min_:
                    min_ = min_val
                if max_val > max_:
                    max_ = max_val
            boundary_dict[param][metric] = (min_, max_)
    return boundary_dict


# ------------------------------------ Load Data and set X - Values ----------------------------------------------------

# Load the W dictionary from the pickle file
with open('SJB_G_Data.pkl', 'rb') as file:
    W_data_dict_01 = pickle.load(file)

# Load the W dictionary from the pickle file
with open('SJB_V_Data.pkl', 'rb') as file:
    W_data_dict_02 = pickle.load(file)

# Load the W dictionary from the pickle file
with open('SJB_N_Data.pkl', 'rb') as file:
    W_data_dict_03 = pickle.load(file)

# define x values
W_x_arr_01 = np.array([1, 1, 2, 2])  # W-Specimens
W_x_arr_02 = np.array([1, 1, 2, 2])  # W-Specimens
W_x_arr_03 = np.array([1, 1, 2, 2])  # W-Specimens
N_x_arr = np.array([300, 350, 370, 400, 440, 500, 600, 650, 700])  # N-Specimens

# N_images, N_curve_params = get_data_matrix_dictionary(N_data_dict, N_x_arr)
W_images_01, W_curve_params_01 = get_data_matrix_dictionary(W_data_dict_01, W_x_arr_01)
W_images_02, W_curve_params_02 = get_data_matrix_dictionary(W_data_dict_02, W_x_arr_02)
W_images_03, W_curve_params_03 = get_data_matrix_dictionary(W_data_dict_03, W_x_arr_03)
images = [W_images_01, W_images_02, W_images_03]

# ---------------------------------------------- plotting --------------------------------------------------------------

metrics = ['R²', 'MAPE', 'CV']

# if N_curve_params == W_curve_params:
#     curve_params = N_curve_params
# else:
#     print('warning not equal curve params!')
#     curve_params = None

curve_params = W_curve_params_01

# N_boundary_dict = get_boundary_dict(curve_params, N_images, metrics)
W_boundary_dict_01 = get_boundary_dict(curve_params, W_images_01, metrics)
W_boundary_dict_02 = get_boundary_dict(curve_params, W_images_02, metrics)
W_boundary_dict_03 = get_boundary_dict(curve_params, W_images_03, metrics)

bound_dict = combine_boundary_dicts([W_boundary_dict_01, W_boundary_dict_02, W_boundary_dict_03], curve_params, metrics)

# plot
for param in curve_params:
    meta_data = [img[param]['meta_data'] for img in images]
    for metric in metrics:
        img_array = [img[param][metric] for img in images]
        height, width = np.shape(img_array[0])
        fig, axes = plt.subplots(2, width, figsize=(14, 9))
        fig.suptitle(param + ' ' + metric)
        fig.subplots_adjust(right=0.9)
        fig.canvas.manager.full_screen_toggle()
        vals = []
        for ii in range(height):
            for jj in range(width):
                ax = axes[ii, jj]
                for m_data in meta_data:
                    ids = m_data['sample_ids']
                    x = m_data['x'][ii, jj, :]
                    y = m_data['y'][ii, jj, :]
                    for ind, id_ in enumerate(ids):
                        ax.text(x[ind], y[ind], id_, fontsize=5, ha='center')

                    ax.scatter(x, y, edgecolors='black', linewidths=0.5)
                val = np.mean(np.array([img_arr[ii, jj] for img_arr in img_array]))  # calc mean value Todo is this ok?
                min_val = bound_dict[param][metric][0]
                max_val = bound_dict[param][metric][1]
                color, color_val = get_normalized_color(min_val, max_val, val)  # get color
                vals.append(val)
                ax.set_facecolor(color)
                ax.tick_params(axis='both', which='major', labelsize=6)
        # Create a normalize object to map values to the 0-1 range
        norm = Normalize(vmin=np.min(np.array(vals)), vmax=np.max(np.array(vals)))
        # Create a ScalarMappable and initialize a color map
        sm = ScalarMappable(norm=norm, cmap='jet')
        # Create a single colorbar for the entire figure
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        # Set the colorbar label
        cbar.set_label(metric)
        fig.subplots_adjust(left=0.03, right=0.91, top=0.94, bottom=0.05, wspace=0.2, hspace=0.2)

        # plt.show()
        fig.savefig(f'figures\\{param}_{metric}.png', dpi=600)
