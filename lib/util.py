import yaml
import json
import requests
import numpy as np
from pyqubo import Array

from . import api_key, dma_api_key, dau_url, dma_url



def post_solve(problem_body: dict) -> dict:
    """Post and solve the problem using Fujitsu DAU API

    The function posts the problem to Fujitsu DAU API. To post and solve the problem, it requires
    the api_key. Please setup the api_key variables inadvance

    In addition to posting the problem to Fujitsu DAU API, the function also registers the job_id to
    the DAU management app to track the use of DAU. 
    This also requires the dma_api_key. Please setup the dma_api_key variables inadvance.
    For futher information about the DAU management app, please refer to the following link:
        https://dau.emath.tw/

    Args:
        problem_body: dict object that contains the problem. 
            For example, the problem_body for the following problem 2x1x2 - 4x2x4 - 3  is:
            {
                "fujitsuDA3": {
                    "time_limit_sec": 10,
                    "penalty_coef" : 1000,
                    "num_run" : 16,
                    "num_group" : 16,
                    "gs_level" : 90,
                    "gs_cutoff" : 90000,
                },
                "binary_polynomial": {
                    "terms": [
                        {"c": 2, "p": [1, 2] },
                        {"c": -4, "p": [2, 4] },
                        {"c": -3, "p": [] }
                    ] 
                }
            }
    
    Return: dict object that contains the job_id.
    """
    problem_header = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Api-Key": api_key
    }

    response = requests.post(
        dau_url+"/da/v3/async/qubo/solve", 
        headers=problem_header, 
        data=json.dumps(problem_body,default=int)
        )
    
    job_id = response.json()['job_id']
    
    register_response = requests.post(dma_url + "/api/posts", headers={
        "Content-Type": "application/json",
        "Accept": "application/json"
    }, data=json.dumps({
     "api_key" : dma_api_key,
     "job_id" : job_id,
     "time_limit_sec" : problem_body["fujitsuDA3"]['time_limit_sec']
    }))
    print(register_response.json())
    
    return response.json()


def get_solution(job_id:str) -> dict:
    """Get the solution of the problem using Fujitsu DAU API

    The function gets the solution of the problem from Fujitsu DAU API. To get the solution, it requires
    the api_key. Please setup the api_key variables inadvance and also give the job_id that is returned by the Fujitsu DAU API.

    Args:
        job_id: string object that contains the job_id.
    
    Return: dict object that contains the solution.
    """
    solution_header = {
        "Job_ID": job_id,
        "Accept": "application/json",
        "X-Api-Key": api_key 
    }
    solution = requests.get(dau_url+"/da/v3/async/jobs/result/"+job_id, headers=solution_header)
    return solution.json()


def get_matrix_term(matrix_element:dict, variables:list) -> list:
    """Get the matrix term from the qubo and variables

    The function is used to get the matrix term from the qubo and variables and encode the matrix term in terms indecies.
    For example, we already have the hamiltonian of the following problem 2x1x2 - 4x2x4 - 3, and the variables are [x1, x2, x4].

    >>> from pyqubo import Binary
    >>> x1 = Binary('x1')
    >>> x2 = Binary('x2')
    >>> x4 = Binary('x4')
    >>> hamiltonian = 2*x1*x2 - 4*x2*x4 - 3
    >>> model = hamiltonian.compile()
    >>> qubo, offset = model.to_qubo()
    >>> get_matrix_term(qubo, model.variables)
    [{'c': 2.0, 'p': [0, 1]}, {'c': -4.0, 'p': [1, 2]}]

    Args:
        matrix_element: dict object that contains the qubo matrix elements.
        variables: list object that contains model's variables.

    Return: list object that contains the matrix term, which is encoded in terms indecies and the coefficients
    """
    matrix_terms = []
    for key, value in matrix_element.items():
        term = {}
        term['c'] = value
        term['p'] = [variables.index(key[0]), variables.index(key[1])]
        matrix_terms.append(term)
    return matrix_terms

def binary_representation(number, precision):
    
    integer, float_number = str(float(number)).split('.')
    integer = int(integer)
    float_number = float('0.' + float_number)
    
    remain = float_number
    
    int_binary_repr_str = bin(integer)[2:]

    int_binary_code = []
    for c in int_binary_repr_str:
        int_binary_code.append(int(c))
    
    float_binary_code = []
    for i in range(1, precision+1):
        binary_power_number = 2**(-i)

        temp = remain - binary_power_number
        if temp >= 0:
            remain = temp
            float_binary_code.append(1)
        else:
            float_binary_code.append(0)
    
    return int_binary_code, float_binary_code


def delete_job(job_id:str) -> dict:
    """Delete the job from the Fujitsu DAU API

    The function deletes the job from the Fujitsu DAU API. To delete the job, it requires
    the api_key. Please setup the api_key variables inadvance and also give the job_id that is returned by the Fujitsu DAU API.

    Args:
        job_id: string object that contains the job_id.

    Return: dict object that contains the response of the delete request. The response would contain the result of
    the computation, if the job is finished. Otherwise, the response would only contain the message. 
    """
    solution_header = {
        "Job_ID": job_id,
        "Accept": "application/json",
        #"X-Access-Token": token
        "X-Api-Key": api_key 
    }
    delete_response = requests.delete(dau_url+"/da/v3/async/jobs/result/"+job_id, headers=solution_header)
    return delete_response.json()

def create_binary_variable_matrix(name, dimension):
    
    A_initial = Array.create(name, shape=dimension, vartype='BINARY')
    A = np.zeros(dimension).tolist()
    
    nrows, ncols = dimension      
    for i in range(nrows):
        for j in range(ncols):
            A[i][j] = A_initial[i][j]
    
    return np.matrix(A)


def create_binary_representated_vector(precision, start_point):
    binary_code = np.zeros(precision)
    pw = start_point
    for i in range(precision):
        binary_code[i] = 2 ** pw
        pw -= 1
    
    return binary_code

def generate_random_matrix(dimension):
    return np.matrix(np.random.rand(dimension[0], dimension[1]) + 1) 

def generate_target_matrix(dimension, precision, start_point):
    binary_code = create_binary_representated_vector(precision, start_point)
    
    # it is declared as a matrix but actually a vector
    # to multiply with matrix
    binary_code_matrix = np.matrix(binary_code).transpose()
    
    columns = []

    nrows, ncols = dimension
    
    for ncol in range(ncols): # nrow means # row <<>> nrows mean the number of rows
        col_matrix = create_binary_variable_matrix('col%d' % ncol, (nrows, precision))
        columns.append(np.array((col_matrix * binary_code_matrix).transpose())[0])
          
        
        
    return np.matrix(np.column_stack(columns))