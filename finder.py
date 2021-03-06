import argparse
import globals
import csv
import random

VAL_FIELD_TYPE = ['speed', 'color', 'distance', 'integral', 'deviation', 'derivative', 'stop', 'step', 'angle']
VAL_SYS_TYPE = ['veh 1', 'veh 2']

PENALTY_TIME = 0.9
MAX_TIME_DIFF = 30

PENALTY_COUPLING = 0.1

COUPLING_MATRIX = [
        [PENALTY_COUPLING, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, PENALTY_COUPLING, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, PENALTY_COUPLING, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, PENALTY_COUPLING, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, PENALTY_COUPLING, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, PENALTY_COUPLING, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, PENALTY_COUPLING, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, PENALTY_COUPLING, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, PENALTY_COUPLING]
    ]



def read_data(data_file):
    file = open(data_file,'r',encoding = 'utf-8-sig')
    data = csv.reader(file)
    const_header = []
    const_values = []
    for ch in next(data):
        if ch: 
            const_header.append(ch)
    for c in next(data):
        if c:
            const_values.append(float(c))
    
    const_len = len(const_header)
    if len(const_header) != len(const_values):
        print('Constant headers and values do not match.')
        const_len = min(len(const_header), len(const_values))
    
    const_dict = dict()
    for i in range(const_len):
        const_dict[const_header[i]] = const_values[i]
    
    var_header = []
    for vh in next(data):
        if vh: 
            var_header.append(vh)
    
    var_data = [[] for i in range(len(var_header))]
    for (j, line) in enumerate(data):
        for i in range(len(var_header)):
            var_data[i].append(float(line[i]))
    return (const_header, const_values, var_header, var_data)

def generate_value(len_const, len_var, len_var_data):
    rand_v_type = random.randrange(len_const+len_var)
    v = globals.Value()
    if rand_v_type < len_var:
        v.type = globals.VAL_TYPE_VAR
        v.index = random.randrange(len_var)
        v.time = random.randrange(MAX_TIME_DIFF)
    else:
        v.type = globals.VAL_TYPE_CONS
        v.index = random.randrange(len_const)
    return v

def generate_proposition(len_const, len_var, len_var_data):
    rand_op = random.randrange(len(globals.OPERATORS))
    op = globals.OPERATORS[rand_op]

    lv = generate_value(len_const, len_var, len_var_data)
    rv = generate_value(len_const, len_var, len_var_data)
    return globals.Prop(lv, op, rv)

def generate_initial_statements(num_population, const_header, var_header, len_var_data):
    population = []

    while len(population) < num_population:
        lp = generate_proposition(len(const_header), len(var_header), len_var_data)
        rp = generate_proposition(len(const_header), len(var_header), len_var_data)

        statement = globals.Statement(0, lp, rp)
        if check_statement(statement):
            population.append(statement)
    return population

def convert_time(statement):
    list_time = []
    if statement.p_left.v_left.type == globals.VAL_TYPE_VAR:
        list_time.append(statement.p_left.v_left.time)
    if statement.p_left.v_right.type == globals.VAL_TYPE_VAR:
        list_time.append(statement.p_left.v_right.time)
    if statement.p_right.v_left.type == globals.VAL_TYPE_VAR:
        list_time.append(statement.p_right.v_left.time)
    if statement.p_right.v_right.type == globals.VAL_TYPE_VAR:
        list_time.append(statement.p_right.v_right.time)
    
    min_time = min(list_time)
    if statement.p_left.v_left.type == globals.VAL_TYPE_VAR:
        statement.p_left.v_left.time -= min_time
    if statement.p_left.v_right.type == globals.VAL_TYPE_VAR:
        statement.p_left.v_right.time -= min_time
    if statement.p_right.v_left.type == globals.VAL_TYPE_VAR:
        statement.p_right.v_left.time -= min_time
    if statement.p_right.v_right.type == globals.VAL_TYPE_VAR:
        statement.p_right.v_right.time -= min_time
    return statement

def calculate_score(statement, const_values, variable_data, penalty_cohesion):
    time_len = []
    if statement.p_left.v_left.type == globals.VAL_TYPE_VAR:
        lp_lv = variable_data[statement.p_left.v_left.index]
        lp_lv_t = statement.p_left.v_left.time
        time_len.append(lp_lv_t)
    else:
        lp_lv = [const_values[statement.p_left.v_left.index]]*len(variable_data[statement.p_left.v_left.index])
        lp_lv_t = 0
    
    if statement.p_left.v_right.type == globals.VAL_TYPE_VAR:
        lp_rv = variable_data[statement.p_left.v_right.index]
        lp_rv_t = statement.p_left.v_right.time
        time_len.append(lp_rv_t)
    else:
        lp_rv = [const_values[statement.p_left.v_right.index]]*len(variable_data[statement.p_left.v_right.index])
        lp_rv_t = 0

    if statement.p_right.v_left.type == globals.VAL_TYPE_VAR:
        rp_lv = variable_data[statement.p_right.v_left.index]
        rp_lv_t = statement.p_right.v_left.time
        time_len.append(rp_lv_t)
    else:
        rp_lv = [const_values[statement.p_right.v_left.index]]*len(variable_data[statement.p_right.v_left.index])
        rp_lv_t = 0
    
    if statement.p_right.v_right.type == globals.VAL_TYPE_VAR:
        rp_rv = variable_data[statement.p_right.v_right.index]
        rp_rv_t = statement.p_right.v_right.time
        time_len.append(rp_rv_t)
    else:
        rp_rv = [const_values[statement.p_right.v_right.index]]*len(variable_data[statement.p_right.v_right.index])
        rp_rv_t = 0

    data_len = [len(lp_lv), len(lp_rv), len(rp_lv), len(rp_rv)]

    min_time = min(time_len)

    max_time = 0
    if lp_lv_t != 0:
        lp_lv_t -= min_time
        max_time = max(max_time, lp_lv_t)
    if lp_rv_t != 0:
        lp_rv_t -= min_time
        max_time = max(max_time, lp_rv_t)
    if rp_lv_t != 0:
        rp_lv_t -= min_time
        max_time = max(max_time, rp_lv_t)
    if rp_rv_t != 0:
        rp_rv_t -= min_time
        max_time = max(max_time, rp_rv_t)
    min_len = min(data_len) - max_time

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(min_len):
        first_left  = lp_lv[i + lp_lv_t]
        first_right = lp_rv[i + lp_rv_t]
        first = globals.OPERATOR_DICT[statement.p_left.op](first_left, first_right)
        second_left = rp_lv[i + rp_lv_t]
        second_right = rp_rv[i + rp_rv_t]
        second = globals.OPERATOR_DICT[statement.p_right.op](second_left, second_right)
        if first and second:
            tp += 1
        elif first and not second:
            tn += 1
        elif not first and second:
            fp += 1
        elif not first and not second:
            fn += 1
        
    if (tp + tn) == 0:
        score = 0
    else: 
        score = float(tp) / float(tp + tn) * 100.0
        score = get_adjusted_score(score, statement, const_header, var_header, penalty_cohesion)
        if max_time >= MAX_TIME_DIFF:
            score = score * (1-PENALTY_TIME)
    return (tp, tn, fp, fn, score)

def evaluate_population(population, const_values, var_data, penalty_cohesion):
    pop_fit = []
    for p in population:
        (tp, tn, fp, fn, score) = calculate_score(p, const_values, var_data, penalty_cohesion)
        pop_fit.append((p, tp, tn, fp, fn, score))
    return pop_fit

def crossover(copy_s1, copy_s2, crossover_rate):
    is_prop = random.random()
    if is_prop < 1.0:
        (o1, o2) = crossover_prop(copy_s1, copy_s2, crossover_rate)
    else:
        (o1, o2) = crossover_val_op(copy_s1, copy_s2, crossover_rate)
    return (o1, o2)

def crossover_prop(copy_s1, copy_s2, crossover_rate):
    is_direct = True
    
    if is_direct:
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_left, copy_s2.p_left) = (copy_s2.p_left, copy_s1.p_left)
        
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_right, copy_s2.p_right) = (copy_s2.p_right, copy_s1.p_right)
    else:
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_left, copy_s2.p_right) = (copy_s2.p_right, copy_s1.p_left)
    
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_right, copy_s2.p_left) = (copy_s2.p_left, copy_s1.p_right)
    return (copy_s1, copy_s2)

def crossover_val_op(copy_s1, copy_s2, crossover_rate):
    keep_prop_order = True
    keep_value_order = True
    
    if keep_prop_order:
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_left.op, copy_s2.p_left.op) = (copy_s2.p_left.op, copy_s1.p_left.op)
        
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_right.op, copy_s2.p_right.op) = (copy_s2.p_right.op, copy_s1.p_right.op)

        if keep_value_order:
            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_left, copy_s2.p_left.v_left) = (copy_s2.p_left.v_left, copy_s1.p_left.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_right, copy_s2.p_left.v_right) = (copy_s2.p_left.v_right, copy_s1.p_left.v_right)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_left, copy_s2.p_right.v_left) = (copy_s2.p_right.v_left, copy_s1.p_right.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_right, copy_s2.p_right.v_right) = (copy_s2.p_right.v_right, copy_s1.p_right.v_right)
        else:
            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_left, copy_s2.p_left.v_right) = (copy_s2.p_left.v_right, copy_s1.p_left.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_right, copy_s2.p_left.v_left) = (copy_s2.p_left.v_left, copy_s1.p_left.v_right)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_left, copy_s2.p_right.v_right) = (copy_s2.p_right.v_right, copy_s1.p_right.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_right, copy_s2.p_right.v_left) = (copy_s2.p_right.v_left, copy_s1.p_right.v_right)
    else:
        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_left.op, copy_s2.p_right.op) = (copy_s2.p_right.op, copy_s1.p_left.op)

        do_crossover = random.random() < crossover_rate
        if do_crossover:
            (copy_s1.p_right.op, copy_s2.p_left.op) = (copy_s2.p_left.op, copy_s1.p_right.op)
        
        if keep_value_order:
            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_left, copy_s2.p_right.v_left) = (copy_s2.p_right.v_left, copy_s1.p_left.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_right, copy_s2.p_right.v_right) = (copy_s2.p_right.v_right, copy_s1.p_left.v_right)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_left, copy_s2.p_left.v_left) = (copy_s2.p_left.v_left, copy_s1.p_right.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_right, copy_s2.p_left.v_right) = (copy_s2.p_left.v_right, copy_s1.p_right.v_right)
        else:
            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_left, copy_s2.p_right.v_right) = (copy_s2.p_right.v_right, copy_s1.p_left.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_left.v_right, copy_s2.p_right.v_left) = (copy_s2.p_right.v_left, copy_s1.p_left.v_right)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_left, copy_s2.p_left.v_right) = (copy_s2.p_left.v_right, copy_s1.p_right.v_left)

            do_crossover = random.random() < crossover_rate
            if do_crossover:
                (copy_s1.p_right.v_right, copy_s2.p_left.v_left) = (copy_s2.p_left.v_left, copy_s1.p_right.v_right)
    return (copy_s1, copy_s2)

def mutate(individual, mutation_rate, len_const, len_var, len_var_data):
    is_prop = random.random()
    if is_prop < 1.0:
        o1 = mutate_prop(individual, mutation_rate, len_const, len_var, len_var_data)
    else:
        o1 = mutate_val_op(individual, mutation_rate, len_const, len_var, len_var_data)
    return o1

def mutate_prop(individual, mutation_rate, len_const, len_var, len_var_data):
    is_replace = True
    if is_replace:
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_left = generate_proposition(len_const, len_var, len_var_data)
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_right = generate_proposition(len_const, len_var, len_var_data)
    else:
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            (individual.p_left, individual.p_right) = (individual.p_right, individual.p_left)
    return individual

def mutate_val_op(individual, mutation_rate, len_const, len_var, len_var_data):
    is_replace = True
    within_prop = True
    keep_value_order = True
    if is_replace:
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            rand_op = random.randrange(len(globals.OPERATORS))
            individual.p_left.op = globals.OPERATORS[rand_op]
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            rand_op = random.randrange(len(globals.OPERATORS))
            individual.p_right.op = globals.OPERATORS[rand_op]
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_left.v_left = generate_value(len_const, len_var, len_var_data)
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_left.v_right = generate_value(len_const, len_var, len_var_data)
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_right.v_left = generate_value(len_const, len_var, len_var_data)
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            individual.p_right.v_right = generate_value(len_const, len_var, len_var_data)
    else:
        do_mutate = random.random() < mutation_rate
        if do_mutate:
            (individual.p_left.op, individual.p_right.op) = (individual.p_right.op, individual.p_left.op)
        if within_prop:
            do_mutate = random.random() < mutation_rate
            if do_mutate:
                (individual.p_left.v_left, individual.p_left.v_right) = (individual.p_left.v_right, individual.p_left.v_left)
            do_mutate = random.random() < mutation_rate
            if do_mutate:
                (individual.p_right.v_left, individual.p_right.v_right) = (individual.p_right.v_right, individual.p_right.v_left)
        else:
            if keep_value_order:
                do_mutate = random.random() < mutation_rate
                if do_mutate:
                    (individual.p_left.v_left, individual.p_right.v_left) = (individual.p_right.v_left, individual.p_left.v_left)
                do_mutate = random.random() < mutation_rate
                if do_mutate:
                    (individual.p_left.v_right, individual.p_right.v_right) = (individual.p_right.v_right, individual.p_left.v_right)
            else:
                do_mutate = random.random() < mutation_rate
                if do_mutate:
                    (individual.p_left.v_left, individual.p_right.v_right) = (individual.p_right.v_right, individual.p_left.v_left)
                do_mutate = random.random() < mutation_rate
                if do_mutate:
                    (individual.p_left.v_right, individual.p_right.v_left) = (individual.p_right.v_left, individual.p_left.v_right)
    return individual

def convert_statement_to_string(prefix, s):
    out = prefix + '\t: (' + s.p_left.v_left.type + '(i=' + str(s.p_left.v_left.index) + ', t=' + str(s.p_left.v_left.time) + ')' + s.p_left.op
    out = out + s.p_left.v_right.type + '(i=' + str(s.p_left.v_right.index) + ', t=' + str(s.p_left.v_right.time) + '),\t'
    out = out + s.p_right.v_left.type + '(i=' + str(s.p_right.v_left.index) + ', t=' + str(s.p_right.v_left.time) + ')' + s.p_right.op
    out = out + s.p_right.v_right.type + '(i=' + str(s.p_right.v_right.index) + ', t=' + str(s.p_right.v_right.time) + '))'
    return out

def convert_statement_with_header(statement, const_header, var_header):
    statement = convert_time(statement)
    retval = []
    if statement.p_left.v_left.type == globals.VAL_TYPE_CONS:
        retval.append(const_header[statement.p_left.v_left.index])
    else:
        retval.append(var_header[statement.p_left.v_left.index] + ' t+' + str(statement.p_left.v_left.time))

    retval.append(statement.p_left.op)

    if statement.p_left.v_right.type == globals.VAL_TYPE_CONS:
        retval.append(const_header[statement.p_left.v_right.index])
    else:
        retval.append(var_header[statement.p_left.v_right.index] + ' t+' + str(statement.p_left.v_right.time))

    if statement.p_right.v_left.type == globals.VAL_TYPE_CONS:
        retval.append(const_header[statement.p_right.v_left.index])
    else:
        retval.append(var_header[statement.p_right.v_left.index] + ' t+' + str(statement.p_right.v_left.time))

    retval.append(statement.p_right.op)

    if statement.p_right.v_right.type == globals.VAL_TYPE_CONS:
        retval.append(const_header[statement.p_right.v_right.index])
    else:
        retval.append(var_header[statement.p_right.v_right.index] + ' t+' + str(statement.p_right.v_right.time))

    return retval

def write_to_csv(population_fitness, const_header, var_header, parameters, index, output_prefix=''):
    file_name = output_prefix + 'evaluation_'+ str(index) +'.csv'
    with open(file_name, 'w', newline='') as csvfile: 
        csv_writer = csv.writer(csvfile) 
        for p in parameters:
            csv_writer.writerow(p)
        csv_writer.writerow(['LPLV', 'LPOP', 'LPRV', 'RPLV', 'RPOP', 'RPRV', 'tp', 'tn', 'fp', 'fn', 'Score'])
        idx = 1
        for m in sorted(population_fitness, key=lambda x: (x[5], x[1]), reverse=True):
            ret = convert_statement_with_header(m[0], const_header, var_header)
            ret.append(m[1])
            ret.append(m[2])
            ret.append(m[3])
            ret.append(m[4])
            ret.append(m[5])
            csv_writer.writerow(ret)
            idx += 1

def check_statement_value_type(statement):
    if statement.p_left.v_left.type == globals.VAL_TYPE_CONS and statement.p_left.v_right.type == globals.VAL_TYPE_CONS:
        return False
    if statement.p_right.v_left.type == globals.VAL_TYPE_CONS and statement.p_right.v_right.type == globals.VAL_TYPE_CONS:
        return False
    return True

def check_statement_duplicate(statement):
    if statement.p_left == statement.p_right:
        return False
    
    if statement.p_left.v_left == statement.p_right.v_right and statement.p_left.v_right == statement.p_right.v_left and statement.p_left.op != statement.p_right.op:
        return False

    if statement.p_left.v_left == statement.p_left.v_right or statement.p_right.v_left == statement.p_right.v_right:
        return False
    return True

def check_statement(statement):
    if not check_statement_value_type(statement):
        return False
    if not check_statement_duplicate(statement):
        return False
    return True

def get_prop_value_names(proposition, const_header, var_header):
    if proposition.v_left.type == globals.VAL_TYPE_CONS:
        lv = const_header[proposition.v_left.index].lower()
    else:
        lv = var_header[proposition.v_left.index].lower()
    if proposition.v_right.type == globals.VAL_TYPE_CONS:
        rv = const_header[proposition.v_right.index].lower()
    else:
        rv = var_header[proposition.v_right.index].lower()
    return (lv, rv)

def get_adjusted_score(raw_score, statement, const_header, var_header,penalty_cohesion):
    left_field_cohesion, left_sys_cohesion = check_prop_cohesion(statement.p_left, const_header, var_header)
    right_field_cohesion, right_sys_cohesion = check_prop_cohesion(statement.p_right, const_header, var_header)

    (is_left_field_cohesive, left_prop_id) = left_field_cohesion
    (is_right_field_cohesive, right_prop_id) = right_field_cohesion

    (is_left_sys_cohesive, left_sys_id) = left_sys_cohesion
    (is_right_sys_cohesive, right_sys_id) = right_sys_cohesion

    score = raw_score
    if not is_left_field_cohesive:
        score = score * (1-penalty_cohesion)
    if not is_right_field_cohesive:
        score = score * (1-penalty_cohesion)
    
    if not is_left_sys_cohesive:
        score = score * (1-penalty_cohesion)
    if not is_right_sys_cohesive:
        score = score * (1-penalty_cohesion)
    
    if is_left_sys_cohesive and is_right_sys_cohesive and left_sys_id == right_sys_id:
        if is_left_field_cohesive and is_right_field_cohesive:
            score = score * (1-check_statement_coupling(left_prop_id, right_prop_id))

    return score

def check_prop_cohesion(proposition, const_header, var_header):
    (lv, rv) = get_prop_value_names(proposition, const_header, var_header)

    left_field_index, left_sys_index = categorize_value(lv)
    right_field_index, right_sys_index = categorize_value(rv)
    
    field_cohesion = (False, 0)
    if left_field_index != -1 and left_field_index == right_field_index:
        field_cohesion = (True, left_field_index)

    sys_cohesion = (False, 0)
    if left_sys_index != -1 and left_sys_index == right_sys_index:
        sys_cohesion = (True, left_sys_index)
    
    return field_cohesion, sys_cohesion

def check_statement_coupling(pl_id, pr_id):
    return COUPLING_MATRIX[pl_id][pr_id]

def categorize_value(value):
    retval_field_index = -1
    for (i, vt) in enumerate(VAL_FIELD_TYPE):
        if vt in value:
            retval_field_index = i

    retval_sys_index = -1
    for (j, st) in enumerate(VAL_SYS_TYPE):
        if st in value:
            retval_sys_index = j

    return retval_field_index, retval_sys_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Belief-Finder')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--output-prefix', required=False)
    parser.add_argument('-i', '--initial-population', default=200, type=int)
    parser.add_argument('-p', '--population-size', default=100, type=int)
    parser.add_argument('-c', '--crossover-rate', default=0.6, type=float)
    parser.add_argument('-m', '--mutation-rate', default=0.2, type=float)
    parser.add_argument('-b', '--budget', default=100, type=int)
    parser.add_argument('-r', '--repeat', default=1, type=int)
    
    args = parser.parse_args()

    data_file = args.data
    num_initial_pop = args.initial_population
    num_population = args.population_size
    crossover_rate = args.crossover_rate
    mutation_rate = args.mutation_rate
    budget = args.budget
    repeat = args.repeat
    
    parameters = []
    parameters.append(['num_initial_pop', str(num_initial_pop)])
    parameters.append(['num_population', str(num_population)])
    parameters.append(['crossover_rate', str(crossover_rate)])
    parameters.append(['mutation_rate', str(mutation_rate)])
    parameters.append(['budget', str(budget)])

    output_prefix = ''
    if args.output_prefix:
        output_prefix = args.output_prefix + '_'

    (const_header, const_values, var_header, var_data) = read_data(data_file)

    for repeat_count in range(repeat):
        penalty_cohesion = 0
        population = generate_initial_statements(num_initial_pop, const_header, var_header, len(var_data[0]))
        population_fitness = evaluate_population(population, const_values, var_data, penalty_cohesion)
        population_fitness = sorted(population_fitness, key=lambda x: (x[5], x[1]), reverse=True)

        idx = 0
        while idx < budget:
            idx += 1
            parent_fitness = population_fitness[:num_population]
            parents = [pf[0] for pf in parent_fitness]

            p_len = len(parents)
            p_half = int(p_len / 2)
            offspring = []

            for i in range(p_len):
                if i >= p_half:
                    break
                copy_s1 = parents[i].copy()
                copy_s2 = parents[i+p_half].copy()
                (o1, o2) = crossover(copy_s1, copy_s2, crossover_rate)
                o1 = mutate(o1, mutation_rate, len(const_values), len(var_data), len(var_data[0]))
                o2 = mutate(o2, mutation_rate, len(const_values), len(var_data), len(var_data[0]))
                if check_statement(o1) and not o1 in parents and not o1 in offspring:
                    offspring.append(o1)

                if check_statement(o2) and not o2 in parents and not o2 in offspring:
                    offspring.append(o2)

            population = parents + offspring
            population_fitness = evaluate_population(population, const_values, var_data, penalty_cohesion)
            population_fitness = sorted(population_fitness, key=lambda x: (x[5], x[1]), reverse=True)
            penalty_cohesion += float(1/budget)
        parent_fitness = population_fitness[:num_population]
        write_to_csv(parent_fitness, const_header, var_header, parameters, repeat_count, output_prefix)
        