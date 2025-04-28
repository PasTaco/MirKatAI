import mysql.connector
from mysql.connector import errorcode # Specific import from the same library
import csv


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
    @staticmethod
    def list_tables(db_conn) -> list[str]:
        """Retrieve the names of all tables in the database."""
        # Include print logging statements so you can see when functions are being called.
        print(' - DB CALL: list_tables()')

        cursor = db_conn.cursor()

        # Fetch the table names.
        cursor.execute("SHOW TABLES;")

        tables = cursor.fetchall()
        return [t[0] for t in tables]

    @staticmethod
    def get_table_schema(db_conn, table_name: str) -> list[tuple[str, str]]:
        """Look up the table schema.

        Returns:
        List of columns, where each entry is a tuple of (column, type).
        """
        print(f' - DB CALL: describe_table({table_name})')

        cursor = db_conn.cursor()

        cursor.execute(f"DESCRIBE `{table_name}`;")
        
        schema = cursor.fetchall()
        # MySQL returns (Field, Type, Null, Key, Default, Extra), so we extract the first two columns.
        return [(col[0], col[1]) for col in schema]

    @staticmethod
    def describe_columns(mirkat_columns_desctiption, table_name:str) -> list[tuple[str,str]]:
        """ Looks for the columns in the table table_name and gets the 
            biological description of the table
            Args:
                table_name (str): Name of the table to describe
            Returns:
                list[tuple[str,str]]: List of tuples containing column names and their descriptions
        """
        # Check if the table name exists in the DataFrame
        if table_name not in mirkat_columns_desctiption['Table'].values:
            print(f"Error: Table '{table_name}' not found.")
            return []

        # Filter the DataFrame for the specified table name
        filtered_df = mirkat_columns_desctiption[mirkat_columns_desctiption['Table'] == table_name]

        # Extract column names and descriptions
        columns = list(zip(filtered_df['Column Name'], filtered_df['Description']))
        
        return columns

    @staticmethod
    def describe_tabes(mirkat_tables_desctiption) -> list[tuple[str,str]]:
        """ Looks for the biological description 
        and returns the description of all the tables
        """
        # Extract table names and descriptions
        tables = list(zip(mirkat_tables_desctiption['Table'], mirkat_tables_desctiption['Description']))
        
        return tables

    @staticmethod
    def execute_query(db_conn, sql: str, query_name:str) -> list[list[str]]:
        """Execute an SQL statement, returning the results.
            params sql: is the formated mySQL query
            params query_name: name of the query only alfanumeric characters.
            """
        print(f' - DB CALL: execute_query({sql})')

        cursor = db_conn.cursor()

        cursor.execute(sql)
        results = cursor.fetchall()
        with open(f"{query_name}.tsv", "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(results) 
        
        return results