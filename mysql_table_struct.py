# utils/mysql_extractor.py
# 获取MySQL数据库的表结构信息和样本数据

import mysql.connector
import pandas as pd

# MySQL 配置（示例）
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'mysql',
    'database': 'prestige_db'
}

tables=['employees','salaries','departments', 'products', 'customers','suppliers']


def get_mysql_summary(config, tables):
    summaries = []
    conn = None
    try:
        conn = mysql.connector.connect(**config)
        for table in tables:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM {table} LIMIT 50")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
            summary = f"[Table: {table}] Row Count: {count}\nColumns: {', '.join(columns)}\nSample Data:\n{df.to_string(index=False)}"
            summaries.append(summary)
    except Exception as e:
        summaries.append(f"[MySQL Error] {str(e)}")
    finally:
        if conn:
            conn.close()
    return "\n\n".join(summaries)


if __name__ == "__main__":
    summary = get_mysql_summary(MYSQL_CONFIG, tables=tables)
    print(summary)
