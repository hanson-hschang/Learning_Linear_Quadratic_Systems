from sim_parameters import *
from sim_plot_result import read_K, calculate_gain_error


memory_length = temp_algo_params["memory_length"]
tau1 = temp_algo_params["tau1"]
tau2 = temp_algo_params["tau2"]
M1 = temp_algo_params["M1"]
M2 = temp_algo_params["M2"]
epsilon1 = temp_algo_params["epsilon1"]
epsilon2 = temp_algo_params["epsilon2"]

signature  = str(memory_length) + "_" + str(tau1) + "_" + str(tau2) + "_" + str(M1) + "_" + str(M2) \
                    + "_" + str(epsilon1) + "_" + str(epsilon2)

initial_val_dir = "initial_values_set" + str(temp_algo_params["initial_values_index"]) + "/" 
benchmark_initial_dir = "initial_values_set" + str(temp_algo_params["initial_values_index"]) + "/benchmark_algo/" 
improved_initial_dir = "initial_values_set" + str(temp_algo_params["initial_values_index"]) + "/improved_algo/" 

if temp_algo_params["exact_inner_loop"]:
    # dir = improved_initial_dir + "exact_inner_sol/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = improved_initial_dir + "exact_inner_sol/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_sol/estimated_outer_grad/"

    else: 
        dir = improved_initial_dir + "exact_inner_sol/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_sol/exact_outer_grad/"

elif temp_algo_params["inner_loop_model_free"]:
    # dir = improved_initial_dir + "estimated_inner_grad/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = improved_initial_dir + "estimated_inner_grad/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "estimated_inner_grad/estimated_outer_grad/"
    else: 
        dir = improved_initial_dir + "estimated_inner_grad/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "estimated_inner_grad/exact_outer_grad/"

else: 
    # dir = improved_initial_dir + "exact_inner_grad/"
    if temp_algo_params["outer_loop_model_free"]:
        dir = improved_initial_dir + "exact_inner_grad/estimated_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_grad/estimated_outer_grad/"
    else: 
        dir = improved_initial_dir + "exact_inner_grad/exact_outer_grad/"
        benchmark_dir = benchmark_initial_dir + "exact_inner_grad/estimated_outer_grad/"


nash_K_filename = "nash/" + "K.pk"
nash_cost_filename = "nash/" + "cost.pk"
nash_lambda_filename = "nash/" + "lambda.pk"
nash_K_file = open(nash_K_filename, "rb")
nash_cost_file = open(nash_cost_filename, "rb")
nash_lambda_file = open(nash_lambda_filename, "rb")
nash_K = pickle.load(nash_K_file)
nash_cost = pickle.load(nash_cost_file)
nash_lambda = pickle.load(nash_lambda_file)
nash_K_array = read_K(nash_K)

from riccati_equation import Riccati
riccati = Riccati(temp_model_params)
matrix_K_time_trajectory = riccati.get_gain_K_time_trajectory()

plt.rcParams['text.usetex'] = True
fontsize = 18
gain_K_fig = plt.figure()
gain_K_ax = gain_K_fig.subplots(1, 1)
cost_fig = plt.figure()
cost_ax = cost_fig.subplots(1,1)

ind_list = np.arange(5)

neuirps_data_folder_directory = "neurips_data/"
if not os.path.isdir(neuirps_data_folder_directory):
    os.mkdir(neuirps_data_folder_directory)

for ind in ind_list:
    print("Loading benchmark file with ind = " + str(ind))
    # benchmark algo list filenames
    benchmark_cost_list_filename = benchmark_dir + "cost_list/" + signature + "_" + str(ind) + ".pk"
    benchmark_cost_diff_filename = benchmark_dir + "cost_diff_list/" + signature + "_" + str(ind) + ".pk"
    benchmark_lambda_list_filename = benchmark_dir + "lambda_list/" + signature + "_" + str(ind) + ".pk"
    benchmark_avg_ng_list_filename = benchmark_dir + "avg_ng_list/" + signature + "_" + str(ind) + ".pk"
    benchmark_KL_list_filename = benchmark_dir + "KL_list/" + signature + "_" + str(ind) + ".pk"
    benchmark_time_list_filename = benchmark_dir + "time_list/" + signature + "_" + str(ind) + ".pk"

    benchmark_cost_list_file = open(benchmark_cost_list_filename, "rb")
    benchmark_cost_diff_file = open(benchmark_cost_diff_filename, "rb")
    benchmark_lambda_list_file = open(benchmark_lambda_list_filename, "rb")
    benchmark_avg_ng_list_file = open(benchmark_avg_ng_list_filename, "rb")
    benchmark_KL_list_file = open(benchmark_KL_list_filename, "rb")
    benchmark_time_list_file = open(benchmark_time_list_filename, "rb")

    benchmark_cost_list = pickle.load(benchmark_cost_list_file)
    benchmark_cost_diff_list = pickle.load(benchmark_cost_diff_file)
    benchmark_lambda_list = pickle.load(benchmark_lambda_list_file)
    benchmark_avg_ng_list = pickle.load(benchmark_avg_ng_list_file)
    benchmark_KL_list = pickle.load(benchmark_KL_list_file)
    benchmark_time_list = pickle.load(benchmark_time_list_file)

    benchmark_K_list = benchmark_KL_list["K_list"]
    benchmark_L_list = benchmark_KL_list["L_list"]
    benchmark_K_error_list = np.zeros(memory_length)
    benchmark_normalized_cost_list = np.array([(x - nash_cost)/nash_cost for x in benchmark_cost_list])

    benchmark_time = np.array(benchmark_time_list)
    benchmark_time = benchmark_time-benchmark_time[0]


    for iter_number in range(memory_length):
        benchmark_K_array = read_K(benchmark_K_list[iter_number])
        benchmark_K_error_list[iter_number] = calculate_gain_error(benchmark_K_array, nash_K_array)

    gain_K_ax.plot(
        benchmark_K_error_list,
        benchmark_time
    )
    cost_ax.plot(
        benchmark_normalized_cost_list,
        benchmark_time
    )

    neuirps_data_file = open(neuirps_data_folder_directory + "Kaiqing_ind_" + str(ind) + ".pk", "wb+")
    pickle.dump(
        dict(
            benchmark_time=benchmark_time,
            benchmark_K_error=benchmark_K_error_list,
            benchmark_normalized_cost=benchmark_normalized_cost_list,
        ),
        neuirps_data_file
    )

gain_K_ax.set_xscale('log')
gain_K_ax.set_xlabel("normalized error", fontsize=fontsize)
gain_K_ax.set_ylabel("time [sec]", fontsize=fontsize)
cost_ax.set_xscale('log')
cost_ax.set_xlabel(r'$(\mathcal{G}(\mathbf{K},\mathbf{L})-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*))/\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=fontsize)
cost_ax.set_ylabel("time [sec]", fontsize=fontsize)


plt.show()