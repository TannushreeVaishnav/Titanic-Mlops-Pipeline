Extract the data i.e titanic.csv from local folder i.e data folder
Transform and load  it  from data local to postgres sql database running on docker container and to view it  we go to dbeaver and connect it then under swl query editior write commant select * from titanic t; 
and see the data as postgres table
next phase is data ingestion using PSYCOPG3