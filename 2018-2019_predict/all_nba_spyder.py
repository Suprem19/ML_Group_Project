
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from bs4 import Comment

import re
import csv
import time

most_current_year = 2020
starting_year = 1977

header_award_list = ['all-nba_1', 'all-nba_2', 'all-nba_3', 'all-defensive_1', 'all-defensive_2', 'all-rookie_1', 'all-rookie_2', 'all_star_game_rosters_1', 'all_star_game_rosters_2']
 
mode = 'per_game'
# mode = 'shooting'

header_per_game = ['Year','Player','Pos','Age','Tm','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
header_shooting = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'MP', 'FG%', 'Dist.', '2P', '0-3', '3-10', '10-16', '16-3P', '3P', '2P', '0-3', '3-10', '10-16', '16-3P', '3P', '2P', '3P', '%FGA', '#', '%3PA', '3P%', 'Att.', '#']


Base_url = "https://www.basketball-reference.com/leagues/NBA_{year}_{mode}.html"
Base_url_award = "https://www.basketball-reference.com/leagues/NBA_{year}.html"

awards_sumary = None

class csv_writer:
    def __init__(self, mode=None):
        
        self.csv_writer = None
        self.header = None
        self.f = None
        csv_filename = 'all_nba_player_{mode}.csv'.format(mode=mode)
        if(mode == 'per_game'):
            self.header = header_per_game + header_award_list
        if(mode == 'shooting'):
            self.header = header_shooting + header_award_list
                    
        # open the file in the write mode
        f= open(csv_filename, 'w', encoding='UTF8', newline='')
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
        

def check_award_for_player(player_name, year, award_info_dic):
    '''
    the value to be writen to csv file should follow oderbelow:
['all-nba_1', 'all-nba_2', 'all-nba_3', 'all-defensive_1', 'all-defensive_2', 'all-rookie_1', 'all-rookie_2', 'all_star_game_rosters_1', 'all_star_game_rosters_2']
'''
    # print(award_info_dic)
    print('cheking year: ', year)
    award_player = ['0']*len(header_award_list)
    if player_name in award_info_dic[year]['all-nba_1']:
        award_player[0] = '1'
    if player_name in award_info_dic[year]['all-nba_2']:
        award_player[1] = '1'
    if (year>= 1989) and (player_name in award_info_dic[year]['all-nba_3']):
        award_player[2] = '1'
    if player_name in award_info_dic[year]['all-defensive_1']:
        award_player[3] = '1'
    if player_name in award_info_dic[year]['all-defensive_2']:
        award_player[4] = '1'
    if player_name in award_info_dic[year]['all-rookie_1']:
        award_player[5] = '1'
    if (year>= 1989) and (player_name in award_info_dic[year]['all-rookie_2']):
        award_player[6] = '1'
    if (year != 1999) and (player_name in award_info_dic[year]['all_star_game_rosters_1']):
        award_player[7] = '1'
    if (year != 1999) and (player_name in award_info_dic[year]['all_star_game_rosters_2']):
        award_player[8] = '1'
    return award_player

def get_player(f, mode):

    for year in range(starting_year, most_current_year + 1):

        phandle = urlopen(Base_url.format(year = year, mode = mode))
        p_soup = BeautifulSoup(phandle, features = "html.parser")

        #Get the header
        #header = [th.getText() for th in p_soup.find_all('tr')[0].find_all('th')]
        #header = header[1:]
        rows = p_soup.find_all('tr')[1:]     
        player_stat = [[td.getText() for td in row.find_all('td')] for row in rows]

        for player in player_stat:
            #There are rows where nothing is stored because they use it as a separator.
            if len(player) == 0:
                continue

            Player_name = player[0]
            Team_name = player[3]
            #We will not store the data if the player is traded mid-season. We store data on individual teams.
            if Team_name == 'TOT':
                continue            

            if mode == 'per_game':
                '''
                Rk	Player	Pos	Age	Tm	G	GS	MP	FG	FGA	FG%	3P	3PA	3P%	2P	2PA	2P%	eFG%	FT	FTA	FT%	ORB	DRB	TRB	AST	STL	BLK	TOV	PF	PTS
                '''
                data = player[:]
                data.insert(0, str(year))
                player_awards = check_award_for_player(Player_name, year, awards_sumary)
                print(player_awards)
                data = data + player_awards
                
            if mode == 'shooting':
                '''
                Rk	Player	Pos	Age	Tm	G	MP	FG%	Dist.		2P	0-3	10-3月	16-10月	16-3P	3P		2P	0-3	10-3月	16-10月	16-3P	3P		2P	3P		%FGA	#		%3PA	3P%		Att.	#
                '''
                data = player[:]
                #There are empty entries where nothing is entered. Get rid of them.
                data[:] = [x for x in data if x]
                data.insert(0,str(year))

            f.write_csv(data)
        time.sleep(1)#to avoid http erro 429

def get_awards():
    award_dic_year = {}
    for year in range(starting_year, most_current_year + 1):
        award_dic = {}
        url = Base_url_award.format(year = year)
        # add header to avoid http erro: 429, 'too many requests'
        # req = Request(url, headers={'User-Agent': 'Mozilla'})
        
        # phandle = urlopen(req)
        phandle = urlopen(url)
        p_soup = BeautifulSoup(phandle, features = "html.parser")

        #div_id: all-nba_1, all-nba_2, all-nba_3, all-defensive_1, all-defensive_2, all-rookie_1, all-rookie_2, all_star_game_rosters_1, all_star_game_rosters_2
        #Find comments

        comments = p_soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            c_soup = BeautifulSoup(c, features = "html.parser")
            for award in header_award_list:
                divs = c_soup.find_all('div', {'id':award})

                for div in divs:
                    players = div.find_all('a')
                    # print(award, '------------------------------------')
                    award_dic[award] = []
                    for player in players:
                        Player_name = player.getText()
                        #Get player_id
                        award_dic[award].append(Player_name)
                   
        award_dic_year[year] = award_dic  
        print("Finished scrape award data for year {}".format(year))
        time.sleep(1)
    return award_dic_year
    

def test_to_find_structure():

    year = 2020
    phandle = urlopen(Base_url_award.format(year = year))
    p_soup = BeautifulSoup(phandle, features = "html.parser")

    #div_id: all-nba_1, all-nba_2, all-nba_3, all-defensive_1, all-defensive_2, all-rookie_1, all-rookie_2, all_star_game_rosters_1, all_star_game_rosters_2

    #Find comments

    comments = p_soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        c_soup = BeautifulSoup(c, features = "html.parser")

        #get the all-nba 1st team
        divs = c_soup.find_all('div', {'id':'all-nba_1'})
        for div in divs:
            players = div.find_all('a')
            for player in players:
                print(player.getText())
                

if __name__ == "__main__":
    # test_to_find_structure()
     awards_sumary = get_awards()
     f = csv_writer(mode)
     get_player(f, mode)
     f.close_csv()
    