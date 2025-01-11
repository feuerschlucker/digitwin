import sqlite3
import pandas as pd


def load_measurement_protocol(db_filepath):

    conn = sqlite3.connect(db_filepath)

    query = f"SELECT * FROM MASTER"
    df = pd.read_sql_query(query, conn)
    conn.close()
    unique_values_dict = {col: df[col].unique().tolist() for col in df.columns}
    for item in unique_values_dict:
        print(item)
    print(unique_values_dict["S_SERIES"])
    return unique_values_dict


def main():
    print("main")
    db_filepath = "/home/heiko/database_cree/repeat_v1.db"
    xxx = load_measurement_protocol(db_filepath)
    
    
    #print(xxx)

if __name__ =="__main__":
    main()