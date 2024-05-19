from sim_parameters import *

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

ind = 0

cost_list_filename = dir + "cost_list/" + signature + "_" + str(ind) + ".pk"
cost_diff_filename = dir + "cost_diff_list/" + signature + "_" + str(ind) + ".pk"
lambda_list_filename = dir + "lambda_list/" + signature + "_" + str(ind) + ".pk"
avg_ng_list_filename = dir + "avg_ng_list/" + signature + "_" + str(ind) + ".pk"

# benchmark algo list filenames
benchmark_cost_list_filename = benchmark_dir + "cost_list/" + signature + "_" + str(ind) + ".pk"
benchmark_cost_diff_filename = benchmark_dir + "cost_diff_list/" + signature + "_" + str(ind) + ".pk"
benchmark_lambda_list_filename = benchmark_dir + "lambda_list/" + signature + "_" + str(ind) + ".pk"
benchmark_avg_ng_list_filename = benchmark_dir + "avg_ng_list/" + signature + "_" + str(ind) + ".pk"

nash_cost_filename = "nash/" + "cost.pk"
nash_lambda_filename = "nash/" + "lambda.pk"

# cost_list_file = open(cost_list_filename, "rb")
# cost_diff_file = open(cost_diff_filename, "rb")
# lambda_list_file = open(lambda_list_filename, "rb")
# avg_ng_list_file = open(avg_ng_list_filename, "rb")

benchmark_cost_list_file = open(benchmark_cost_list_filename, "rb")
benchmark_cost_diff_file = open(benchmark_cost_diff_filename, "rb")
benchmark_lambda_list_file = open(benchmark_lambda_list_filename, "rb")
benchmark_avg_ng_list_file = open(benchmark_avg_ng_list_filename, "rb")

benchmark_cost_file = open(nash_cost_filename, "rb")
benchmark_lambda_file = open(nash_lambda_filename, "rb")

# temp_cost_list = pickle.load(cost_list_file)
# temp_cost_diff_list = pickle.load(cost_diff_file)
# temp_lambda_list = pickle.load(lambda_list_file)
# temp_avg_ng_list = pickle.load(avg_ng_list_file)

benchmark_cost_list = pickle.load(benchmark_cost_list_file)
benchmark_cost_diff_list = pickle.load(benchmark_cost_diff_file)
benchmark_lambda_list = pickle.load(benchmark_lambda_list_file)
benchmark_avg_ng_list = pickle.load(benchmark_avg_ng_list_file)

nash_cost = pickle.load(benchmark_cost_file)
nash_lambda = pickle.load(benchmark_lambda_file)

compare_plot = 1

# Already plotted the linear convergence part
plt.rcParams['text.usetex'] = True
# plot linear convergence
plt.figure()
if compare_plot:
    # plt.plot(temp_cost_diff_list, label="Our algo")
    plt.plot(benchmark_cost_diff_list, linestyle='--', label="Benchmark algo")
    plt.yscale('log')
    plt.xlabel('Outer-loop iterations $t$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}(\mathbf{K}_t))-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.legend()
    plt.savefig(dir + signature + "/compare_cost_diff.pdf")
    
else: 
    # plt.plot(temp_cost_diff_list, label='Our algo')
    plt.yscale('log')
    plt.xlabel('Outer-loop iterations $t$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}(\mathbf{K}_t))-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    
    plt.savefig(dir + signature + "/cost_diff.pdf")





# plot cost plot
plt.figure()
if compare_plot:

    # plt.plot([x - nash_cost for x in temp_cost_list], label="Our algo")
    plt.plot([x - nash_cost for x in benchmark_cost_list], linestyle='--', label="Benchmark algo")
    # plt.axhline(y=nash_cost, linestyle='--', label=r'$\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$')
    plt.yscale('log')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}_k)-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)
    
    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/compare_cost.pdf")
else:

    # plt.plot(temp_cost_list, label='Our algorithm')
    # plt.axhline(y=nash_cost, linestyle='--', label=r'$\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\mathcal{G}(\mathbf{K}_t,\mathbf{L}_k)-\mathcal{G}(\mathbf{K}^*,\mathbf{L}^*)$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/cost.pdf")
    



# plot lambda plot
plt.figure()
if compare_plot:

    # plt.plot([nash_lambda - x for x in temp_lambda_list], label="Our algo")
    plt.plot([nash_lambda - x for x in benchmark_lambda_list], linestyle='--', label="Benchmark algo")
    # plt.axhline(y=nash_lambda, linestyle='--', label=r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})$')
    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$', fontsize=18)
    plt.ylabel(r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})-\lambda_{\min}(\mathbf{H}_{\mathbf{K}_t,\mathbf{L}_k })$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/compare_lambda.pdf")

else: 

    # plt.plot(nash_lambda - temp_lambda_list, label='Our algorithm')
    # plt.axhline(y=nash_lambda, linestyle='--', label=r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})$')

    plt.xlabel(r'Total iterations of $(\mathbf{K}_t,\mathbf{L}_k)$')
    plt.ylabel(r'$\lambda_{\min}(\mathbf{H}_{\mathbf{K}^*,\mathbf{L}^*})-\lambda_{\min}(\mathbf{H}_{\mathbf{K}_t,\mathbf{L}_k })$', fontsize=18)

    plt.legend()
    if not os.path.isdir(dir + signature):
        os.mkdir(dir + signature)
    plt.savefig(dir + signature + "/lambda.pdf")



# plot average gradient norm plot
plt.figure()
plt.yscale('log')
if compare_plot:

    # plt.plot(temp_avg_ng_list, label="Our algo")
    plt.plot(benchmark_avg_ng_list, linestyle='--', label="Benchmark algo")
    plt.legend()
    plt.xlabel(r'Outer-loop iterations $t$', fontsize=18)
    plt.ylabel('Average natural gradient norm', fontsize=18)
    
    plt.savefig(dir + signature + "/compare_avg_ng_norm.pdf")
else: 

    # plt.plot(temp_avg_ng_list)
    plt.xlabel(r'Outer-loop iterations $t$', fontsize=18)
    plt.ylabel('Average natural gradient norm', fontsize=18)
    # plt.legend(ncol=1, loc='upper right')

    # plt.savefig(dir + signature + "/avg_ng_norm.pdf", bbox="tight")
    plt.savefig(dir + signature + "/avg_ng_norm.pdf")