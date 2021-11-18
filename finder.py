import argparse
import xml.etree.ElementTree as ET
import globals
import csv
import random


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
    
    # var_dict = dict()
    # for i in range(len(var_header)):
    #     var_dict[var_header[i]] = var_data[i]
    # return (const_dict, var_dict)
    return (const_values, var_data)

def generate_initial_statements(num_population, len_const, len_var, len_var_data):
    population = []

    for i in range(num_population):
        for j in range(2):
            rand_op = random.randrange(6)
            op = globals.OPERATORS[rand_op]

            rand_lv_type = random.randrange(len_const+len_var)
            lv = globals.Value()
            if rand_lv_type < len_var:
                lv.type = globals.VAL_TYPE_VAR
                lv.index = random.randrange(len_var)
                lv.time = random.randrange(len_var_data)
            else:
                lv.type = globals.VAL_TYPE_CONS
                lv.index = random.randrange(len_const)

            rand_rv_type = random.randrange(len_const+len_var)
            rv = globals.Value()
            if rand_rv_type < len_var:
                rv.type = globals.VAL_TYPE_VAR
                rv.index = random.randrange(len_var)
                rv.time = random.randrange(len_var_data)
            else:
                rv.type = globals.VAL_TYPE_CONS
                rv.index = random.randrange(len_const)
            
            if j == 0:
                lp = globals.Prop(lv, op, rv)
            else:
                rp = globals.Prop(lv, op, rv)
            
        statement = globals.Statement(i, lp, rp)
        population.append(statement)
        
    # def generate_initial_statements(num_population, const_dict, var_dict):
    #     const_len = len(const_dict)
    #     var_len = len(var_dict)
    #     len_var_data = len(list(var_dict.values())[0])
        
    #     population = []

    #     for i in range(num_population):
    #         for j in range(2):
    #             rand_op = random.randrange(6)
    #             op = globals.OPERATORS[rand_op]

    #             rand_lv_type = random.randint(0, 1)
    #             lv = globals.Value()
    #             if rand_lv_type == 0:
    #                 lv.type = globals.VAL_TYPE_VAR
    #                 lv.index = random.randrange(var_len)
    #                 lv.time = random.randrange(len_var_data)
    #             else:
    #                 lv.type = globals.VAL_TYPE_CONS
    #                 lv.index = random.randrange(const_len)

    #             rand_rv_type = random.randint(0, 1)
    #             rv = globals.Value()
    #             if rand_lv_type == 0:
    #                 rv.type = globals.VAL_TYPE_VAR
    #                 rv.index = random.randrange(var_len)
    #                 rv.time = random.randrange(len_var_data)
    #             else:
    #                 rv.type = globals.VAL_TYPE_CONS
    #                 rv.index = random.randrange(const_len)
                
    #             if j == 0:
    #                 lp = globals.Prop(lv, op, rv)
    #             else:
    #                 rp = globals.Prop(lv, op, rv)
                
    #         statement = globals.Statement(i, lp, rp)
    #         population.append(statement)
    #     return population

    return population

def calculate_score(statement, const_values, variable_data):
    if statement.p_left.v_left.type == globals.VAL_TYPE_VAR:
        lp_lv = variable_data[statement.p_left.v_left.index]
        lp_lv_t = statement.p_left.v_left.time
    else:
        lp_lv = [const_values[statement.p_left.v_left.index]]*len(variable_data[statement.p_left.v_left.index])
        lp_lv_t = 0
    
    if statement.p_left.v_right.type == globals.VAL_TYPE_VAR:
        lp_rv = variable_data[statement.p_left.v_right.index]
        lp_rv_t = statement.p_left.v_right.time
    else:
        lp_rv = [const_values[statement.p_left.v_right.index]]*len(variable_data[statement.p_left.v_right.index])
        lp_rv_t = 0

    if statement.p_right.v_left.type == globals.VAL_TYPE_VAR:
        rp_lv = variable_data[statement.p_right.v_left.index]
        rp_lv_t = statement.p_right.v_left.time
    else:
        rp_lv = [const_values[statement.p_right.v_left.index]]*len(variable_data[statement.p_right.v_left.index])
        rp_lv_t = 0
    
    if statement.p_right.v_right.type == globals.VAL_TYPE_VAR:
        rp_rv = variable_data[statement.p_right.v_right.index]
        rp_rv_t = statement.p_right.v_right.time
    else:
        rp_rv = [const_values[statement.p_right.v_right.index]]*len(variable_data[statement.p_right.v_right.index])
        rp_rv_t = 0

    data_len = [len(lp_lv), len(lp_rv), len(rp_lv), len(rp_rv)]
    time_len = [lp_lv_t, lp_rv_t, rp_lv_t, rp_rv_t]

    min_len = min(data_len) - max(time_len)

    tp = 0 # true positive
    tn = 0 # true negative
    fp = 0 # false positive
    fn = 0 # false negative
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
    
    return (tp, tn, fp, fn, score)

def evaluate_population(population, const_values, var_data):
    pop_fit = []
    for p in population:
        (tp, tn, fp, fn, score) = calculate_score(p, const_values, var_data)
        pop_fit.append((p, tp, tn, fp, fn, score))
    return pop_fit

def crossover(population_fitness):
    p_len = len(population_fitness)
    p_half = int(p_len / 2)
    offspring = []

    is_prop_crossover = True

    for i in range(p_len):
        if i >= p_half:
            break
        copy_s1 = population_fitness[i][0].copy()
        copy_s2 = population_fitness[i+p_half][0].copy()

        # Proposition crossover
        if is_prop_crossover:
            (o1, o2) = crossover_proposition(copy_s1, copy_s2)
        else:
            (o1, o2) = crossover_val_op(copy_s1, copy_s2)
        offspring.append(o1)
        offspring.append(o2)
    return offspring

def crossover_proposition(copy_s1, copy_s2):
    single = True
    is_direct = True
    if single:
        if is_direct:
            is_left = random.randint(0, 1)
            if is_left:
                tmp = copy_s1.p_left
                copy_s1.p_left = copy_s2.p_left
                copy_s2.p_left = tmp
            else:
                tmp = copy_s1.p_right
                copy_s1.p_right = copy_s2.p_right
                copy_s2.p_right = tmp
        else:
            is_left_first = random.randint(0, 1)
            if is_left_first:
                tmp = copy_s1.p_left
                copy_s1.p_left = copy_s2.p_right
                copy_s2.p_right = tmp
            else:
                tmp = copy_s1.p_right
                copy_s1.p_right = copy_s2.p_left
                copy_s2.p_left = tmp
    else:
        if is_direct:
            tmp = copy_s1.p_left
            copy_s1.p_left = copy_s2.p_left
            copy_s2.p_left = tmp
            tmp = copy_s1.p_right
            copy_s1.p_right = copy_s2.p_right
            copy_s2.p_right = tmp
        else:
            tmp = copy_s1.p_left
            copy_s1.p_left = copy_s2.p_right
            copy_s2.p_right = tmp
            tmp = copy_s1.p_right
            copy_s1.p_right = copy_s2.p_left
            copy_s2.p_left = tmp
    return (copy_s1, copy_s2)

def crossover_val_op(copy_s1, copy_s2):
    op_direct = True
    val_direct = True
    single_point_op = True

    if single_point_op:
        is_left = random.randint(0, 1)
        if op_direct:
            if is_left:
                tmp = copy_s1.p_left.op
                copy_s1.p_left.op = copy_s2.p_left.op
                copy_s2.p_left.op = tmp
            else:
                tmp = copy_s1.p_right.op
                copy_s1.p_right.op = copy_s2.p_right.op
                copy_s2.p_right.op = tmp
        else:
            if is_left:
                tmp = copy_s1.p_left.op
                copy_s1.p_left.op = copy_s2.p_right.op
                copy_s2.p_right.op = tmp
            else:
                tmp = copy_s1.p_right.op
                copy_s1.p_right.op = copy_s2.p_left.op
                copy_s2.p_left.op = tmp
    else:
        # Operators
        if op_direct:
            tmp = copy_s1.p_left.op
            copy_s1.p_left.op = copy_s2.p_left.op
            copy_s2.p_left.op = tmp
            tmp = copy_s1.p_right.op
            copy_s1.p_right.op = copy_s2.p_right.op
            copy_s2.p_right.op = tmp
        else:
            tmp = copy_s1.p_left.op
            copy_s1.p_left.op = copy_s2.p_right.op
            copy_s2.p_right.op = tmp
            tmp = copy_s1.p_right.op
            copy_s1.p_right.op = copy_s2.p_left.op
            copy_s2.p_left.op = tmp
    
    cross_point = random.randrange(4)
    if val_direct:
        if cross_point == 0:
            tmp = copy_s1.p_left.v_left
            copy_s1.p_left.v_left = copy_s2.p_left.v_left
            copy_s2.p_left.v_left = tmp
        elif cross_point == 1:
            tmp = copy_s1.p_left.v_right
            copy_s1.p_left.v_right = copy_s2.p_left.v_right
            copy_s2.p_left.v_right = tmp
        elif cross_point == 2:
            tmp = copy_s1.p_right.v_left
            copy_s1.p_right.v_left = copy_s2.p_right.v_left
            copy_s2.p_right.v_left = tmp
        else:
            tmp = copy_s1.p_right.v_right
            copy_s1.p_right.v_right = copy_s2.p_right.v_right
            copy_s2.p_right.v_right = tmp
    else:
        if cross_point == 0:
            tmp = copy_s1.p_left.v_left
            copy_s1.p_left.v_left = copy_s2.p_right.v_left
            copy_s2.p_right.v_left = tmp
        elif cross_point == 1:
            tmp = copy_s1.p_left.v_right
            copy_s1.p_left.v_right = copy_s2.p_right.v_right
            copy_s2.p_right.v_right = tmp
        elif cross_point == 2:
            tmp = copy_s1.p_right.v_left
            copy_s1.p_right.v_left = copy_s2.p_left.v_left
            copy_s2.p_left.v_left = tmp
        else:
            tmp = copy_s1.p_right.v_right
            copy_s1.p_right.v_right = copy_s2.p_left.v_right
            copy_s2.p_left.v_right = tmp
    return (copy_s1, copy_s2)

def convert_statement_to_string(prefix, s):
    out = prefix + '\t: (' + s.p_left.v_left.type + '(i=' + str(s.p_left.v_left.index) + ', t=' + str(s.p_left.v_left.time) + ')' + s.p_left.op
    out = out + s.p_left.v_right.type + '(i=' + str(s.p_left.v_right.index) + ', t=' + str(s.p_left.v_right.time) + '),\t'
    out = out + s.p_right.v_left.type + '(i=' + str(s.p_right.v_left.index) + ', t=' + str(s.p_right.v_left.time) + ')' + s.p_right.op
    out = out + s.p_right.v_right.type + '(i=' + str(s.p_right.v_right.index) + ', t=' + str(s.p_right.v_right.time) + '))'
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher-Order-Belief-Mutation')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--output-prefix', required=False)
    parser.add_argument('-i', '--initial-population', default=200, type=int)
    parser.add_argument('-p', '--population-size', default=100, type=int)
    parser.add_argument('-c', '--crossover-rate', default=0.2, type=float)
    parser.add_argument('-m', '--mutation-rate', default=0.2, type=float)
    parser.add_argument('-b', '--budget', default=10000, type=int)
    
    args = parser.parse_args()

    data_file = args.data
    num_initial_pop = args.initial_population
    num_population = args.population_size
    crossover_rate = args.crossover_rate
    mutation_rate = args.mutation_rate
    budget = args.budget

    output_prefix = ''
    if args.output_prefix:
        output_prefix = args.output_prefix + '_'

    
    (const_values, var_data) = read_data(data_file)
    # (const_dict, var_dict) = read_data(data_file)
    population = generate_initial_statements(num_initial_pop, len(const_values), len(var_data), len(var_data[0]))
    # population = generate_initial_statements(num_population, const_dict, var_dict)

    population_fitness = evaluate_population(population, const_values, var_data)
    population_fitness = sorted(population_fitness, key=lambda x: (x[5], x[1]), reverse=True)

    idx = 0
    while idx < 10:
        idx += 1
        parent_fitness = population_fitness[:num_population]
        best = parent_fitness[0]
        print('Best Fitness: {0}, tp: {2}\n\t{1}'.format(best[5], convert_statement_to_string('', best[0]), best[1]))

        offspring = crossover(parent_fitness)
        offspring_fitness = evaluate_population(offspring, const_values, var_data)
        population_fitness = parent_fitness + offspring_fitness
        population_fitness = sorted(population_fitness, key=lambda x: (x[5], x[1]), reverse=True)