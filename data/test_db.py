import sqlite3

conn = sqlite3.connect('./data/mlb_reentrenamiento.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tablas = cursor.fetchall()
print("Tablas encontradas en la DB:", tablas)
conn.close()