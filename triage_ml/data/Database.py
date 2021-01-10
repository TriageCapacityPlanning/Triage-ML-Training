import psycopg2
from typing import Text, List, Tuple


class Database:

    def __init__(self, dbname: Text, user: Text, password: Text, host: Text, port: int):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def get_table_data(self, table: Text, columns: List[Text]) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT (%s) FROM %s', ','.join(columns), table)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def close(self):
        self.conn.close()
