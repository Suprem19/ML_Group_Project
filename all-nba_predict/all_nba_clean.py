# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:54:41 2022

@author: maths
"""
import re
import csv
import time

header_award_list = ['all-nba_1', 'all-nba_2', 'all-nba_3', 'all-defensive_1', 'all-defensive_2', 'all-rookie_1', 'all-rookie_2', 'all_star_game_rosters_1', 'all_star_game_rosters_2']
 


header_per_game = ['Year','Player','Pos','Age','Tm','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
header_shooting = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'FG%', 'Dist.', '2P', '0-3', '3-10', '10-16', '16-3P', '3P', '2P', '0-3', '3-10', '10-16', '16-3P', '3P', '2P', '3P', '%FGA', '#', '%3PA', '3P%', 'Att.', '#']
header_advanced = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']

mode = 'per_game'

# csv_filename_original = 'all_nba_player_per_game_train.csv'
# csv_filename_cleaned = 'all_nba_player_per_game_train_cleaned.csv'

csv_filename_original = 'all_nba_player_per_game_test.csv'
csv_filename_cleaned = 'all_nba_player_per_game_test_cleaned.csv'


class csv_writer:
    def __init__(self, mode=None):
        
        self.csv_writer = None
        self.header = None
        self.f = None
        if(mode == 'per_game'):
            self.header = header_per_game + header_award_list
        if(mode == 'shooting'):
            self.header = header_shooting + header_award_list
        if(mode == 'advanced'):
            self.header = header_advanced + header_award_list 
            
        # open the file in the write mode
        f= open(csv_filename_cleaned, 'w', encoding='UTF8', newline='')
        self.f = f
        
        # create the csv writer
        self.csv_writer = csv.writer(f)
        # write a row to the csv file
        self.csv_writer.writerow(self.header )
        
    def write_csv(self, data):
        # print('writing data: ', data)
        self.csv_writer.writerow(data)
    
    def close_csv(self):
        try:
            self.f.close()
            print('csv file closed.')

        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
                         # but may be overridden in exception subclasses


    
def check_transifer_impact(csv_filename):

    with  open(csv_filename, encoding='UTF8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        year_last_row = None
        player_name_last_row = None
        
        for row in csv_reader:
            if(row[0] == year_last_row) and (row[1] == player_name_last_row):
                #duplicated player
                if '1' in row[-9:]:
                    print(row[0], row[1], row[-9:])
               
            year_last_row = row[0]
            player_name_last_row = row[1]

def process_use_pandas():
      import pandas
      #https://stackoverflow.com/questions/17666075/python-pandas-groupby-result
      df = pandas.read_csv(csv_filename_original, header = 0)
      # df['G'] = df.groupby(['Year', 'Player'])['G'].transform('sum')
      c = df.groupby(['Year', 'Player'])
      print(c.groups, dir(c))
      df.drop_duplicates()
      df.to_csv(csv_filename_cleaned)


def merge_row(tmp_row, current_row):
    if len(tmp_row) ==0:
        return current_row
    else:
        # SUM 'G' number of Games
        # print('points are ', tmp_row[5], current_row[5])
        tmp_row[5] = str(int(tmp_row[5]) + int(current_row[5]))
     
        # AVG from 7 to 29
        for i in range(7,30):
            if len(tmp_row[i]) == 0:
                tmp_row[i] = '0'
            if len(current_row[i]) == 0:
                current_row[i] = '0'
            tmp_row[i] = str((float(tmp_row[i]) + float(current_row[i]))/2)
        return tmp_row
        
if __name__ == "__main__":
    # test_to_find_structure()
  
    f_new = csv_writer(mode)
    with open(csv_filename_original, encoding='UTF8', newline='') as f:
        year_last_row = None
        player_name_last_row = None
        csv_rd = csv.reader(f, delimiter=',')
        row_index =0
        csv_list = []
        header_flag = 0
        for row in csv_rd:  
            row.append(0)# append last element to indicate  duplicate
            if(row[0] == year_last_row) and (row[1] == player_name_last_row):
                row[-1] = 1 #indicate duplication
                print(row_index)
                csv_list[row_index-1][-1]=1
                #duplicated player
                if '1' in row[-9:]:
                    #dupilicated player is in the all-nba/all-defender/all-star
                    print(row[0], row[1], row[-9:])
                    
            year_last_row = row[0]
            player_name_last_row = row[1]
            if header_flag==1:
                header_flag =0
            else:
                row_index = row_index +1
                csv_list.append(row)
        
        merged_row = []
        row_index =0
        
        for row in csv_list:            
            # print(row_index, row)
            if row[-1] == 1: 
                # print('next row: ', row_index+1)
                print(row_index +1, csv_list[row_index +1])
                
                merged_row = merge_row(merged_row, row)
                if csv_list[row_index +1][-1] ==0:
                    # print('merged_row', merged_row)
                    f_new.write_csv(merged_row[:-1]) 
            else:
                merged_row =[]
                f_new.write_csv(row[:-1])
                
            row_index = row_index +1
            
    f_new.close_csv()
      


      