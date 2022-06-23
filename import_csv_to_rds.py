import json
import mysql.connector
import os
import csv

connection = mysql.connector.connect(host='database-1.co3pho0xsqet.us-east-1.rds.amazonaws.com',
                                         database='demo',
                                         port='3306',
                                         user='admin',
                                         passwd='India123')

mysql_insert = "insert into new_hotel(id, dates, occupancy_rate, adr, ori, ari, revpar, rgi, country, market_segment, distribution_channel) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"


#Read the .csv files from folder
with open('/edited_historical_data.csv') as f:
    lines = f.readlines()


cursor = connection.cursor(mysql_insert)

def read_row(lines, mysql_insert):
    try:
        
        for i in lines:
            row = i
            row = row.replace('\n', '')
            #print("type of row", type(row) )
            row_list = (row.split(','))
            #print("row_list =", row_list)
            #create fields
            id = row_list[0]
            dates=row_list[1]
            occupancy_rate=row_list[2]
            adr=row_list[3]
            ori=row_list[4]
            ari=row_list[5]
            revpar=row_list[6]
            rgi=row_list[7]
            country=row_list[8]
            market_segment=row_list[9]
            distribution_channel=row_list[10]
            
            
            #row_tuple = (id, dates, country, market_segment, distribution_channel,occupancy_rate, adr)
            row_tuple = (id, dates, occupancy_rate, adr, ori, ari, revpar, rgi, country, market_segment, distribution_channel  )
            
            cursor.execute(mysql_insert, row_tuple)
            print("Inserting data [id]= {}".format(id))
            connection.commit()
        print("Insert Complete!")
    except Exception as e:
        print("Error in inserting ...{}".format(e))
        
read_row(lines, mysql_insert)

