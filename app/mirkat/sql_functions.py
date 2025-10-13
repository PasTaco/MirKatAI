import mysql.connector
from mysql.connector import errorcode # Specific import from the same library
import csv
import logging


class MySqlConnection:
    def __init__(self, config):
        """
        Initialize the MySqlConnection with a configuration dictionary.
        :param config: A dictionary containing database connection parameters.
        """
        self.config = config
        self.connection = None

    def connect_sql(self):
        """
        Establish a connection to the MySQL database.
        """
        cnx=None
        try:
            cnx = mysql.connector.connect(**self.config)
            return cnx
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            return None




class DBTools:
    def __init__(self, db_conn, mirkat_tables_desctiption, mirkat_columns_desctiption):
        """
        Initialize the DBTools with a database connection and table descriptions.
        :param db_conn: A MySQL database connection object.
        :param mirkat_tables_desctiption: DataFrame containing table descriptions.
        :param mirkat_columns_desctiption: DataFrame containing column descriptions.
        """
        self.db_conn = db_conn
        self.mirkat_tables_desctiption = mirkat_tables_desctiption
        self.mirkat_columns_desctiption = mirkat_columns_desctiption


    def list_tables(self) -> list[str]:
        """Retrieve the names of all tables in the database."""
        # Include print logging statements so you can see when functions are being called.
        print(' - DB CALL: list_tables()')
        logging.info(' - DB CALL: list_tables()')

        cursor = self.db_conn.cursor()

        # Fetch the table names.
        cursor.execute("SHOW TABLES;")

        tables = cursor.fetchall()
        return [t[0] for t in tables]

    def get_table_schema(self, table_name: str) -> list[tuple[str, str]]:
        """Look up the table schema.

        Returns:
        List of columns, where each entry is a tuple of (column, type).
        """
        print(f' - DB CALL: describe_table({table_name})')
        logging.info(f' - DB CALL: describe_table({table_name})')
        cursor = self.db_conn.cursor()

        cursor.execute(f"DESCRIBE `{table_name}`;")
        
        schema = cursor.fetchall()
        # MySQL returns (Field, Type, Null, Key, Default, Extra), so we extract the first two columns.
        return [(col[0], col[1]) for col in schema]

    def describe_columns(self, table_name:str) -> list[tuple[str,str]]:
        """ Looks for the columns in the table table_name and gets the 
            biological description of the table
            Args:
                table_name (str): Name of the table to describe
            Returns:
                list[tuple[str,str]]: List of tuples containing column names and their descriptions
        """
        print(f' - DB CALL: describe_columns({table_name})')
        logging.info(f' - DB CALL: describe_columns({table_name})')
        # Check if the table name exists in the DataFrame
        if table_name not in self.mirkat_columns_desctiption['Table'].values:
            print(f"Error: Table '{table_name}' not found.")
            return []

        # Filter the DataFrame for the specified table name
        filtered_df = self.mirkat_columns_desctiption[self.mirkat_columns_desctiption['Table'] == table_name]

        # Extract column names and descriptions
        columns = list(zip(filtered_df['Column Name'], filtered_df['Description']))
        
        return columns

    def describe_tables(self) -> list[tuple[str,str]]:
        """ Looks for the biological description 
        and returns the description of all the tables
        """
        print(' - DB CALL: describe_tables()')
        logging.info(' - DB CALL: describe_tables()')
        # Extract table names and descriptions
        tables = list(zip(self.mirkat_tables_desctiption['Table'], self.mirkat_tables_desctiption['Description']))
        
        return tables

    def execute_query(self, sql: str, query_name:str) -> list[list[str]]:
        """Execute an SQL statement, returning the results.
            params sql: is the formated mySQL query
            params query_name: name of the query only alfanumeric characters.
            """
        print(f' - DB CALL: execute_query({sql})')
        logging.info(f' - DB CALL: execute_query({sql})')
        self.db_conn.ping(reconnect=True)  # Reconnect if the connection is lost
        cursor = self.db_conn.cursor()

        cursor.execute(sql)
        results = cursor.fetchall()
        print(f"Results from SQL are: {results}")
        with open(f"{query_name}.tsv", "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(results) 
        # if results is too large, get the first 100 rows
        if len(results) > 100:
            results = results[:100]
        cursor.close()
        return results