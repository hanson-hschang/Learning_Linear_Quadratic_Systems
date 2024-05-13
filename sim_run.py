from sim_parameters import *
import time
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    filename="sim_run.log", 
    filemode="w",
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def Simulation(model_params, algo_params):
    logging.info("Start simulation with benchmark_algo = " + str(temp_algo_params["benchmark_algo"] ))
    
    model = ZO_LQ_model(model_params, algo_params)
    
    inner_loop_model_free = int(algo_params["inner_loop_model_free"])
    outer_loop_model_free = int(algo_params["outer_loop_model_free"])
    exact_inner_loop = int(algo_params["exact_inner_loop"])
    benchmark_algo = int(algo_params["benchmark_algo"])
    K = [None for _ in range(model.H)]
    L = [None for _ in range(model.H)]
    for h in range(model.H):
        K[h] = torch.clone(algo_params["K0"][h])
        L[h] = torch.clone(algo_params["L0"][h])
    
    tau1 = algo_params["tau1"]
    tau2 = algo_params["tau2"]
    # to compare with Kaiqing's work, we also choose the cost and the \lambda_{\min}(R^w - D^TPD)
    memory_length = algo_params["memory_length"]
    initial_values_index = algo_params["initial_values_index"]
    cost_list = [np.nan for _ in range(memory_length)]
    cost_diff_list = [np.nan for _ in range(memory_length)]
    lambda_list = [np.nan for _ in range(memory_length)]

    K_list = [np.nan for _ in range(memory_length)]
    L_list = [np.nan for _ in range(memory_length)]
    time_list= [np.nan for _ in range(memory_length)]

    # we also record the convergence measure - average of the gradient, only for the outer-loop iterations
    avg_ng_list = [np.nan for _ in range(memory_length)]
    
    logging.info("Building files ...")

    initial_val_dir = "initial_values_set" + str(initial_values_index) + "/" 
    
    if not os.path.isdir(initial_val_dir):
        os.mkdir(initial_val_dir)

    if benchmark_algo: 
        initial_val_dir += "benchmark_algo/"
    else:
        initial_val_dir += "improved_algo/"
    
    if not os.path.isdir(initial_val_dir):
        os.mkdir(initial_val_dir)

    # nash is irrelevant to the initial values
    if not os.path.isdir("nash/"):
        os.mkdir("nash/")

    nash_K_filename = "nash/K.pk"
    nash_K_file = open(nash_K_filename, "wb+")
    pickle.dump(model.nash_K, nash_K_file)
    nash_L_filename = "nash/L.pk"
    nash_L_file = open(nash_L_filename, "wb+")
    pickle.dump(model.nash_L, nash_L_file)
    nash_cost_filename = "nash/cost.pk"
    nash_cost_file = open(nash_cost_filename, "wb+")
    pickle.dump(model.nash_cost, nash_cost_file)
    nash_lambda_filename = "nash/lambda.pk"
    nash_lambda_file = open(nash_lambda_filename, "wb+")
    pickle.dump(model.nash_lambda, nash_lambda_file)


    logging.info("Start computing ...")
    cost_list[0] = model.compute_cost(K, L)
    cost_diff_list[0] = cost_list[0] - model.nash_cost
    lambda_list[0] = model.compute_lambda(K, L)
    K_list[0] = K.copy()
    L_list[0] = L.copy()
    # avg_ng_list record natural gradients at (K_t, L(K_t)) - only update at outer-loop iterations
    avg_ng_list[0] = model.compute_avg_ng(K, 1, 0)
    # iteration flag
    iter_num = 0
    outer_loop_iter_num = 0

    # for t_out in range(T_outer):
    # print("memory length: ", memory_length)
    time_list[0] = time.time()
    while avg_ng_list[outer_loop_iter_num] > model.epsilon2 and iter_num < memory_length-1:
        # for t_in in range(T_inner):
        
        # initialize L with zeros everytime
        L = [None for _ in range(model.H)]
        for h in range(model.H):
            L[h] = torch.clone(algo_params["L0"][h])
        
        temp_optL = model.compute_optL(K)
        temp_cost = model.compute_cost(K, L)
        temp_optimal_cost = model.compute_cost(K, temp_optL)

        cost_diff = temp_optimal_cost - temp_cost
       
        while cost_diff > model.epsilon1 and iter_num < memory_length-2:
            
            if not exact_inner_loop:

                if inner_loop_model_free:
                    L_ngrad = model.est_L_ngrad(K, L)
                else: 
                    L_ngrad = model.compute_L_ngrad(K, L)
                
                for h in range(model.H):
                    L[h] = L[h] + tau1 * L_ngrad[h] 
            else:
                L = model.compute_optL(K)

            ''' 
            record convergence measure for inner-loop problem
            '''
            iter_num += 1
            temp_cost = model.compute_cost(K, L)
            cost_diff = temp_optimal_cost - temp_cost
            cost_list[iter_num] = temp_cost
            logging.debug("inner loop iteration:" + str(iter_num) + ": " + str(temp_cost))
            logging.debug("cost diff: " + str(cost_diff))
            lambda_list[iter_num] = model.compute_lambda(K, L)
            K_list[iter_num] = K.copy()
            L_list[iter_num] = L.copy()
            time_list[iter_num] = time.time()

        # now L is the estimated inner-loop solution given K
        if outer_loop_model_free:
            if benchmark_algo:
                K_ngrad = model.est_K_ngrad_benchmark(K, L)
            else:
                K_ngrad = model.est_K_ngrad(K, L)    
        
        else:
            K_ngrad = model.compute_K_ngrad(K, L)

        for h in range(model.H):
            K[h] = K[h] - tau2 * K_ngrad[h]

        iter_num += 1
        outer_loop_iter_num += 1
        ''' 
        record convergence measure
        '''
        cost_list[iter_num] = model.compute_cost(K, L)
        cost_diff_list[outer_loop_iter_num] = cost_list[iter_num] - model.nash_cost
        lambda_list[iter_num] = model.compute_lambda(K, L)
        K_list[iter_num] = K.copy()
        L_list[iter_num] = L.copy()
        time_list[iter_num] = time.time()

        avg_ng_list[outer_loop_iter_num] = model.compute_avg_ng(K, outer_loop_iter_num+1, avg_ng_list[outer_loop_iter_num-1])
        logging.debug("outer loop cost: " + str(iter_num) + ": " + str(cost_list[iter_num]))
        logging.debug("outer loop lambda: " + str(iter_num) + ": " + str(lambda_list[iter_num]))
        logging.debug("avg ng grad: " + str(avg_ng_list[outer_loop_iter_num]))

    dir = ""

    if exact_inner_loop:
        if not os.path.isdir(initial_val_dir + "exact_inner_sol/"):
            os.mkdir(initial_val_dir + "exact_inner_sol/")
        if outer_loop_model_free:
            if not os.path.isdir(initial_val_dir + "exact_inner_sol/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_sol/estimated_outer_grad/")
            dir = initial_val_dir + "exact_inner_sol/estimated_outer_grad/"
        else: 
            if not os.path.isdir(initial_val_dir + "exact_inner_sol/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_sol/exact_outer_grad/")
            dir = initial_val_dir + "exact_inner_sol/exact_outer_grad/"
            
    elif inner_loop_model_free:
        if not os.path.isdir(initial_val_dir + "estimated_inner_grad/"):
            os.mkdir(initial_val_dir + "estimated_inner_grad/")
        if outer_loop_model_free:
            if not os.path.isdir(initial_val_dir + "estimated_inner_grad/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "estimated_inner_grad/estimated_outer_grad/")
            dir = initial_val_dir + "estimated_inner_grad/estimated_outer_grad/"
        else: 
            if not os.path.isdir(initial_val_dir + "estimated_inner_grad/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "estimated_inner_grad/exact_outer_grad/")
            dir = initial_val_dir + "estimated_inner_grad/exact_outer_grad/"

    else: 
        if not os.path.isdir(initial_val_dir + "exact_inner_grad/"):
            os.mkdir(initial_val_dir + "exact_inner_grad/")
        if outer_loop_model_free: 
            if not os.path.isdir(initial_val_dir + "exact_inner_grad/estimated_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_grad/estimated_outer_grad/")
            dir = initial_val_dir + "exact_inner_grad/estimated_outer_grad/"
        else:
            if not os.path.isdir(initial_val_dir + "exact_inner_grad/exact_outer_grad/"):
                os.mkdir(initial_val_dir + "exact_inner_grad/exact_outer_grad/")
            dir = initial_val_dir + "exact_inner_grad/exact_outer_grad/"
        
    
    signature  = str(memory_length) + "_" + str(tau1) + "_" + str(tau2) + "_" + str(model.M1) + "_" + str(model.M2) \
                    + "_" + str(model.epsilon1) + "_" + str(model.epsilon2)
    cost_list_filename = dir + "/cost_list/" + signature + "_" + str(0) + ".pk"
    ind = 0
    while os.path.isfile(cost_list_filename):
        ind += 1
        cost_list_filename = dir + "/cost_list/" + signature + "_" + str(ind) + ".pk"
    print("signatrue_ind: " + signature + "_" + str(ind))


    lambda_list_filename = dir + "/lambda_list/" + signature + "_" + str(ind) + ".pk"
    avg_ng_list_filename = dir + "/avg_ng_list/" + signature + "_" + str(ind) + ".pk"
    cost_diff_list_filename = dir + "/cost_diff_list/" + signature + "_" + str(ind) + ".pk"
    KL_list_filename = dir + "/KL_list/" + signature + "_" + str(ind) + ".pk"
    time_list_filename = dir + "/time_list/" + signature + "_" + str(ind) + ".pk"

    if not os.path.isdir(dir + "/cost_list/"):
        os.mkdir(dir + "/cost_list/")
    if not os.path.isdir(dir + "/cost_diff_list/"):
        os.mkdir(dir + "/cost_diff_list/")
    if not os.path.isdir(dir + "/lambda_list/"):
        os.mkdir(dir + "/lambda_list/")
    if not os.path.isdir(dir + "/avg_ng_list/"):
        os.mkdir(dir + "/avg_ng_list/")
    if not os.path.isdir(dir + "/KL_list/"):
        os.mkdir(dir + "/KL_list/")
    if not os.path.isdir(dir + "/time_list/"):
        os.mkdir(dir + "/time_list/")


    cost_file = open(cost_list_filename, "wb+")
    cost_diff_file = open(cost_diff_list_filename, "wb+")
    lambda_file = open(lambda_list_filename, "wb+")
    avg_ng_file = open(avg_ng_list_filename, "wb+")
    KL_file = open(KL_list_filename, "wb+")
    time_file = open(time_list_filename, "wb+")

    pickle.dump(cost_list, cost_file)
    pickle.dump(cost_diff_list, cost_diff_file)
    pickle.dump(lambda_list, lambda_file)
    pickle.dump(avg_ng_list, avg_ng_file)
    pickle.dump(
        dict(
            K_list=K_list,
            L_list=L_list
        ),
        KL_file
    )
    pickle.dump(time_list, time_file)
    logging.debug("debug cost list: " + str(cost_list))

print("runnning sims ...")
# temp_algo_params["benchmark_algo"] = 0
# res = Simulation(temp_model_params, temp_algo_params) 
temp_algo_params["benchmark_algo"] = 1
benchmark_res = Simulation(temp_model_params, temp_algo_params)