import sqlite3
import pandas as pd
from tqdm import tqdm
import os
import shutil
import json


def pack_files(directory, output_filename):
    with open(output_filename, 'wb') as output_file:
        # Iterate through all the files in the specified directory
        for filename in tqdm(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                # Write the filename and file content to the output file
                with open(filepath, 'rb') as f:
                    content = f.read()
                    # Store the filename length and filename
                    output_file.write(len(filename).to_bytes(2, 'little'))
                    output_file.write(filename.encode())
                    # Store the content length and content
                    output_file.write(len(content).to_bytes(8, 'little'))
                    output_file.write(content)


def unpack_files(input_filename, output_directory):
    print('start_unpack')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    with open(input_filename, 'rb') as input_file:
        while True:
            filename_length_bytes = input_file.read(2)
            if not filename_length_bytes:
                break
            filename_length = int.from_bytes(filename_length_bytes, 'little')
            filename = input_file.read(filename_length).decode()
            content_length = int.from_bytes(input_file.read(8), 'little')
            content = input_file.read(content_length)
            with open(os.path.join(output_directory, filename), 'wb') as f:
                f.write(content)
    print('finished unpack')


def load_measurement_protocol(db_filepath):
    # Connect to your database
    conn = sqlite3.connect(db_filepath)
    measurement_protocol = pd.read_sql_query(f"SELECT * FROM 'mp_induction_active'", conn)
    measurement_protocol_dict = measurement_protocol.set_index('MEAS_ID').T.to_dict('list')
    # add names to data
    data_names = ["Sample_ID", "Yoke", "Paste", "Amplitude", "Waveform", "Frequency", "Duration", "Sampling_Rate"]
    for mea_id in measurement_protocol_dict.keys():
        measurement_protocol_dict[mea_id] = dict(zip(data_names, measurement_protocol_dict[mea_id]))
    conn.close()
    return measurement_protocol_dict


def convert_db_to_induct(db_filepath, induct_filepath):
    if not os.path.exists("temp_ind_data"):
        os.makedirs("temp_ind_data")
    #
    measure_dict = load_measurement_protocol(db_filepath)
    with open("temp_ind_data/02_measurement_dict.json", 'w') as file:
        json.dump(measure_dict, file, indent=4)

    conn = sqlite3.connect(db_filepath)
    query = f"""SELECT * FROM data_temperature"""
    df = pd.read_sql_query(query, conn)
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.set_index('Time', inplace=True)
    df.to_hdf(f"temp_ind_data/01_Temp_data.h5", key="data", mode="w")

    for key in tqdm(measure_dict.keys()):
        query = f"""SELECT Time_abs, Cha, ChB, ChC, ChD FROM data_induction_active WHERE MEAS_ID = '{key}'"""
        df = pd.read_sql_query(query, conn)
        df['Time_abs'] = pd.to_datetime(df['Time_abs'], unit='ns')
        df.set_index('Time_abs', inplace=True)
        df.to_hdf(f"temp_ind_data/{key}_data.h5", key="data", mode="w")

    conn.close()
    # 
    pack_files('temp_ind_data', induct_filepath)
    # delete temp folder
    shutil.rmtree("temp_ind_data")


if __name__ == "__main__":
    convert_db_to_induct("stahl_judenburg_29_07_2024.db", 'stahl_judenburg_29_07_2024.induct')
#
