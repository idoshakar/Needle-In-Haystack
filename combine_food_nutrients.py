import duckdb

def init_database(database: str):
    con = duckdb.connect(database)
    con.execute('''
        SET preserve_insertion_order=false;
        SET threads=1;
        CREATE OR REPLACE TABLE food AS
        SELECT fdc_id, description as food_name
            FROM read_csv('./FoodData_Central_csv_2025-04-24/food.csv');

    CREATE OR REPLACE TABLE food_nutrient AS
        SELECT fdc_id, nutrient_id, amount
            FROM read_csv('./FoodData_Central_csv_2025-04-24/food_nutrient.csv');

    CREATE OR REPLACE TABLE nutrient AS
        SELECT id as nutrient_id, name as nutrient_name, unit_name
            FROM read_csv('./FoodData_Central_csv_2025-04-24/nutrient.csv');

    CREATE OR REPLACE TABLE full_join AS
        SELECT food.fdc_id as fdc_id, food_nutrient.nutrient_id as nutrient_id, food_name, nutrient_name, amount, unit_name
            FROM food
                INNER JOIN food_nutrient
                    ON food.fdc_id = food_nutrient.fdc_id
                INNER JOIN nutrient
                    ON food_nutrient.nutrient_id = nutrient.nutrient_id;
    ''')

    con.close()

def build(database: str):
    # Connect to a database (in-memory or persistent)
    con = duckdb.connect(database)
    # Execute the SQL commands
    nutrient_names_dataframe = con.execute('''
        SET preserve_insertion_order=false;
        SET threads=1;

    CREATE OR REPLACE TABLE chosen_nutrients AS
        SELECT nutrient_name, count(*) as c
                FROM full_join
            GROUP BY nutrient_name
        ORDER BY c DESC
        LIMIT 50;

    SELECT * FROM chosen_nutrients
    ''').df()

    #print(nutrient_names_dataframe)
    nutrient_fstr = ''
    for nutrient_name in nutrient_names_dataframe['nutrient_name']:
        nutrient_name = nutrient_name.replace("'", "''")
        nutrient_fstr += 'MAX(CASE WHEN nutrient_name = \''+nutrient_name+'\' THEN amount END) AS "'+nutrient_name+'",'

    nutrient_fstr = nutrient_fstr[:-1]

    con.execute(f'''
    CREATE OR REPLACE TABLE pivoted_full_join AS
            SELECT fdc_id, food_name, {nutrient_fstr}
                FROM full_join
        GROUP BY fdc_id, food_name
    ''')

    description = con.execute('DESCRIBE pivoted_full_join').df()

    print(description)

    # Close the connection (if persistent database)
    con.close()

if __name__ == '__main__':
    build('food_central.db')
