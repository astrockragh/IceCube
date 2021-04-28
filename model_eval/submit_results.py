import pandas as pd
import numpy as np
import argparse
import os
import sqlite3
import sqlalchemy
import time

parser = argparse.ArgumentParser()
parser.add_argument("-list_databases", "--list_databases", type=bool, required=False)
parser.add_argument("-database", "--database", type=str, required=False)
parser.add_argument("-result", "--result", type=str, required=False)
parser.add_argument("-model", "--model", type=str, required=False)
parser.add_argument("-init", "--init", type=str, required=False)
args = parser.parse_args()

start_time = time.time()

def SubmitResults(path, results, init, model_name, accepted_vars,max_write_size):
    print('The Following Variables Have Been Identified:')
    provided_vars = results.columns
    submitted_vars = []
    tablename = init + '_' + model_name
    exists_already = False
    file_check = os.path.isfile(path + '/predictions.db')

    if 'event_no' in provided_vars:
        for accepted_var in accepted_vars:
            if accepted_var in provided_vars:
                submitted_vars.append(accepted_var)    
        count = 0
        for submitted_var in submitted_vars:
            print(submitted_var)
            if count == 0:
                if submitted_var == 'event_no':
                    query_columns =  submitted_var + ' INTEGER PRIMARY KEY NOT NULL' 
                else: 
                    query_columns =  submitted_var + ' FLOAT'
            else:
                if submitted_var == "event_no":
                    query_columns = query_columns + ', ' + submitted_var + ' INTEGER PRIMARY KEY NOT NULL' 
                else:
                    query_columns =  query_columns + ', ' + submitted_var + ' FLOAT'

            count +=1
        if count > 1:
            #CODE = f"PRAGMA foreign_keys=off;\nCREATE TABLE {tablename} ({query_columns});\nPRAGMA foreign_keys=on;"
            CODE = "PRAGMA foreign_keys=off;\nCREATE TABLE {} ({});\nPRAGMA foreign_keys=on;".format(tablename,query_columns) 
        else:
            print('ERROR: Not enough variables present for submission. Are you following the naming convention?')
        
    else:
        print('ERROR: event_no not present in provided data. Please follow the naming convention!')
    conn = sqlite3.connect(path + '/predictions.db')
    c = conn.cursor()
    c.executescript(CODE)
    c.close()
  
    
    
    
    
    submitted_results = results.loc[:,submitted_vars]
    
    n_writes = int(np.ceil(len(submitted_results)/max_write_size))
    writes_list = np.array_split(submitted_results,n_writes)
    print('Submitting to %s in %s writes!'%(tablename,n_writes))
    
    for i in range(0,len(writes_list)):
        engine_main = sqlalchemy.create_engine('sqlite:///'+ path + '/predictions.db')
        writes_list[i].to_sql(tablename,engine_main,index= False, if_exists = 'append')
        engine_main.dispose()
        print('%s / %s'%(i + 1, n_writes))

    print('Submission Complete! Time Elapsed: %s [s]'%(time.time() - start_time))

    return

############ CONFIGURATION ################
supported_databases = ['dev_lvl7_mu_nu_e_classification_v003','IC8611_oscNext_003_final']
accepted_vars = ['event_no','energy_log10_pred','position_x_pred','position_y_pred',
                'position_z_pred','azimuth_pred','zenith_pred','pid_pred',
                'energy_log10_sigma', 'position_x_sigma','position_y_sigma',
                'position_z_sigma', 'azimuth_sigma','zenith_sigma']

general_db_path = '/groups/hep/pcs557/databases/%s/predictions'%args.database
max_write_size = 50000




###########################################



if args.list_databases:
    print('-----------------------------------------------------------')
    print('THE FOLLOWING DATABASES ARE CURRENTLY ACCEPTING PREDICTIONS')
    print('-----------------------------------------------------------')
    for db in supported_databases:
        print(db)
    print('-----------------------------------------------------------')
    
if args.database in supported_databases and args.list_databases != True:
    print('-----------------------------------------------------------')
    print('PLEASE MAKE SURE THAT YOUR SUBMISSION OBEY \n \n \
        1) Your predictions have the same units and scaling as the truth table in %s \n \n \
        2) You have not by mistake predicted on events that your model has trained on'%args.database)
    answer = input('Type "yes" to continue \n')
    possible_answers = ['yes', 'YES','Yes']
    if answer in possible_answers:
        os.makedirs(general_db_path, exist_ok = True)
        try:
            results = pd.read_csv(args.result)
        
        except:
            print('ERROR: Your results could not be loaded. Is the path correct?')
            
        if (len(args.init) + len(args.model)) > 50:
            print('ERROR: Your initials and model name exceeds the maximum character limit of 50!')
        else:        
            SubmitResults(general_db_path, results, args.init, args.model, accepted_vars,max_write_size)
    else:
        print('Cancelling submission!')
    
elif args.list_databases != True:
    print('ERROR: %s currently not supported!. Use --list_databases True to see list of available databases.'%args.database)

