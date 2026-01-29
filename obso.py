import re
import math
import numpy as np
import pandas as pd 

from tqdm import tqdm 
from scipy import signal

import loader as io
import pitch_control as pc

def calc_obso(PPCF, Transition, Score, ball_start_pos, attack_direction=0):
    # calculate obso in single frame
    # PPCF, Score : 50 * 32
    # Transitino : 100 * 64

    Transition = np.array((Transition))
    Score = np.array((Score))
    ball_grid_x = int((ball_start_pos[0]+ 52.5) // (105/50))
    ball_grid_y = int((ball_start_pos[1]+ 34) // (68/32))
    
    # When out of the pitch
    if ball_grid_x < 0:
        ball_grid_x = 0
    elif ball_grid_x > 49:
        ball_grid_x = 49
    if ball_grid_y < 0:
        ball_grid_y = 0
    elif ball_grid_y > 31:
        ball_grid_y = 31
    
    Transition = Transition[31-ball_grid_y:63-ball_grid_y, 49-ball_grid_x:99-ball_grid_x]
    
    if attack_direction < 0:
        Score = np.fliplr(Score)
    elif attack_direction > 0:
        Score = Score
    else:
        print("input attack direction is 1 or -1")

    obso = PPCF * Transition * Score

    return obso, PPCF, Transition, Score


def calc_player_evaluate(player_pos, evaluation):
    # player_pos:(x, y) col
    # evaluation : evaluation grid (32 * 50). ex) obso, ppcf

    # grid size
    grid_size_x = 105 / 50
    grid_size_y = 68 / 32
    
    player_grid_x = (player_pos[0] + 52.5) // grid_size_x
    player_grid_y = (player_pos[1] + 34) // grid_size_y

    # When out of the pitch
    if player_grid_x < 0:
        player_grid_x = 0
    elif player_grid_x > 49:
        player_grid_x = 49
    if player_grid_y < 0:
        player_grid_y = 0
    elif player_grid_y > 31:
        player_grid_y = 31
    
    # data format int in grid number
    player_grid_x = int(player_grid_x)
    player_grid_y = int(player_grid_y)
    
    # be careful for index number (y cordinate, x cordinate)
    player_ev = evaluation[player_grid_y, player_grid_x]

    return player_ev

# calculate player evaluate at event
def calc_player_evaluate_event(OBSO, events, tracking_home, tracking_away):
    # calculate player evaluation at event
    # input:obso(grid evaluation), events(event data in Metrica format), tracking home and away (tracking data)
    # return home_obso, away_obso(player evaluation at event)

    # set DataFrame column name
    column_name = ['event_number', 'event_frame']
    home_column = tracking_home.columns
    home_player_num = [s[:-2] for s in home_column if re.match('Home_\d*_x', s)]
    home_column_name = column_name + home_player_num
    away_column = tracking_away.columns
    away_player_num = [s[:-2] for s in away_column if re.match('Away_\d*_x', s)] 
    away_column_name = column_name + away_player_num
    home_index = list(range(len(events[events['team']=='Home'])))
    away_index = list(range(len(events[events['team']=='Away'])))
    home_obso = pd.DataFrame(columns=home_column_name, index=home_index)
    away_obso = pd.DataFrame(columns=away_column_name, index=away_index)

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0

    for event_num, row in enumerate(events.itertuples()):
        if row.team == 'Home':
            home_event_num += 1
            home_obso['event_frame'].iloc[home_event_num-1] = row.start_frame
            home_obso['event_number'].iloc[home_event_num-1] = event_num

            for player in home_player_num:
                #home_player_pos = [tracking_home[player+'_x'].iloc[frame], tracking_home[player+'_y'].iloc[frame]]
                home_player_pos = [tracking_home[player+'_x'].loc[row.start_frame], tracking_home[player+'_y'].loc[row.start_frame]]
                if not np.isnan(home_player_pos[0]):
                    home_obso[player].iloc[home_event_num-1] = calc_player_evaluate(home_player_pos, OBSO[event_num])
                else:
                    continue
        elif row.team == 'Away':
            away_event_num += 1
            away_obso['event_frame'].iloc[away_event_num-1] = row.start_frame
            away_obso['event_number'].iloc[away_event_num-1] = event_num
            for player in away_player_num:
                away_player_pos = [tracking_away[player+'_x'].loc[row.start_frame], tracking_away[player+'_y'].loc[row.start_frame]]
                if not np.isnan(away_player_pos[0]):
                    away_obso[player].iloc[away_event_num-1] = calc_player_evaluate(away_player_pos, OBSO[event_num])
                else:
                    continue
        else:
            continue

    return home_obso, away_obso

# caculate player evaluate at trace
def calc_player_evaluate_trace(OBSO, tracking_home, tracking_away):
    # calculate player evaluation at event
    # input:obso(grid evaluation), events(event data in Metrica format), tracking home and away (tracking data)
    # return home_obso, away_obso(player evaluation at event)

    # set DataFrame column name
    column_name = ['trace_number', 'trace_frame']
    home_column = tracking_home.columns
    home_player_num = [s[:-2] for s in home_column if re.match('Home_\d*_x', s)]
    home_column_name = column_name + home_player_num
    away_column = tracking_away.columns
    away_player_num = [s[:-2] for s in away_column if re.match('Away_\d*_x', s)] 
    away_column_name = column_name + away_player_num
    home_index = list(range(len(tracking_home)))
    away_index = list(range(len(tracking_away)))
    home_obso = pd.DataFrame(columns=home_column_name, index=home_index)
    away_obso = pd.DataFrame(columns=away_column_name, index=away_index)

    # initialize event number in home and away
    home_trace_num = 0
    away_trace_num = 0

    for trace_num, row in enumerate(tracking_home.itertuples()):
        if row.team == 'Home':
            home_trace_num += 1
            home_obso['trace_frame'].iloc[home_trace_num-1] = row.Index
            home_obso['trace_number'].iloc[home_trace_num-1] = trace_num

            for player in home_player_num:
                home_player_pos = [tracking_home[player+'_x'].loc[row.Index], tracking_home[player+'_y'].loc[row.Index]]
                if not np.isnan(home_player_pos[0]):
                    home_obso[player].iloc[home_trace_num-1] = calc_player_evaluate(home_player_pos, OBSO[trace_num])
                else:
                    continue
        elif row.team == 'Away':            
            away_trace_num += 1
            away_obso['trace_frame'].iloc[away_trace_num-1] = row.Index
            away_obso['trace_number'].iloc[away_trace_num-1] = trace_num
            for player in away_player_num:
                away_player_pos = [tracking_away[player+'_x'].loc[row.Index], tracking_away[player+'_y'].loc[row.Index]]
                if not np.isnan(away_player_pos[0]):
                    away_obso[player].iloc[away_trace_num-1] = calc_player_evaluate(away_player_pos, OBSO[trace_num])
                else:
                    continue
        else:
            continue

    return home_obso, away_obso

# caculate player evaluate at virtual
def calc_player_evaluate_virtual(team, OBSO, tracking_home, tracking_away):
    # calculate player evaluation at event
    # input:obso(grid evaluation), events(event data in Metrica format), tracking home and away (tracking data)
    # return home_obso, away_obso(player evaluation at event)

    if team == 'Home':
        tracking_data = tracking_home
    elif team == 'Away':
        tracking_data = tracking_away
    else:
        raise ValueError(f"Invalid team: {team}")

    # set DataFrame column name
    column_name = ['trace_number', 'trace_frame']
    column = tracking_data.columns
    player_num = [s[:-2] for s in column if re.match(f'{team}_\d*_x', s)]

    index = list(range(len(tracking_data)))
    obso_df = pd.DataFrame(columns=column_name + player_num, index=index)

    for trace_num, row in enumerate(tracking_data.itertuples()):
        obso_df['trace_frame'].iloc[trace_num] = row.Index
        obso_df['trace_number'].iloc[trace_num] = trace_num

        for player in player_num:
            player_pos = [tracking_data[player+'_x'].loc[row.Index], tracking_data[player+'_y'].loc[row.Index]]
            if not np.isnan(player_pos[0]):
                obso_df[player].iloc[trace_num] = calc_player_evaluate(player_pos, OBSO[trace_num])
            else:
                continue

    return obso_df

def calc_onball_obso(events, tracking_home, tracking_away, home_obso, away_obso):
    # calculate on-ball obso because obso is not defined in on-ball 
    # input : event data in format Metrica
    # output : home_onball_obso and away_onball_obso in format pandas dataframe

    # set dataframe column name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]

    # set output dataframe
    home_onball_obso = pd.DataFrame(columns=home_obso.columns, index=list(range(len(home_obso))))
    away_onball_obso = pd.DataFrame(columns=away_obso.columns, index=list(range(len(away_obso))))

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0

    # search on ball player
    for num, frame in enumerate(tqdm(events['start_frame'])):
        if events['team'].iloc[num] == 'Home':
            home_event_num += 1
            dis_dict = {}
            home_onball_obso['event_frame'].iloc[home_event_num-1] = frame
            home_onball_obso['event_number'].iloc[home_event_num-1] = num
            for name in home_name:
                if np.isnan(tracking_home[name+'_x'].loc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array([tracking_home[name+'_x'].loc[frame], tracking_home[name+'_y'].loc[frame]])
                    ball_pos = np.array([tracking_home['ball_x'].loc[frame], tracking_home['ball_y'].loc[frame]])
                    ball_dis = np.linalg.norm(player_pos-ball_pos)
                    dis_dict[name] = ball_dis
            # home onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            home_onball_obso[onball_player].iloc[home_event_num-1] = home_obso[onball_player].iloc[home_event_num-1] 
        elif events['team'].iloc[num] == 'Away':
            away_event_num += 1
            dis_dict = {}
            away_onball_obso['event_frame'].iloc[away_event_num-1] = frame
            away_onball_obso['event_number'].iloc[away_event_num-1] = num
            for name in away_name:
                if np.isnan(tracking_away[name+'_x'].loc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array([tracking_away[name+'_x'].loc[frame], tracking_away[name+'_y'].loc[frame]])
                    ball_pos = np.array([tracking_away['ball_x'].loc[frame], tracking_away['ball_y'].loc[frame]])
                    ball_dis = np.linalg.norm(player_pos-ball_pos)
                    dis_dict[name] = ball_dis
            # away onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            away_onball_obso[onball_player].iloc[away_event_num-1] = away_obso[onball_player].iloc[away_event_num-1]
        else:
            continue
    
    return home_onball_obso, away_onball_obso
        
def remove_offside_obso(events, tracking_home, tracking_away, home_obso, away_obso):
    # remove obso value(to 0) for offise player
    # events:event data (Metrica format), tracking home and away:tracking data (Metrica foramat) 
    # obso (home and away): obso value in each event
    
    # set parameters for calculating PPCF
    params = pc.default_model_params()
    GK_numbers = [io.find_goalkeeper(tracking_home), io.find_goalkeeper(tracking_away)]
    # set player name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]
    # search offside player
    for event_id in tqdm(range(len(events))):
        # check event team home or away 
        if events['team'].iloc[event_id] == 'Home':
            _, _, _, attacking_players = pc.generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, GK_numbers)
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = home_name ^ attacking_players_name
            for name in off_players:
                home_obso[name][home_obso['event_number']==event_id] = 0
        elif events['team'].iloc[event_id] == 'Away':
            _, _, _, attacking_players = pc.generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, GK_numbers)
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = away_name ^ attacking_players_name
            for name in off_players:
                away_obso[name][away_obso['event_number']==event_id] = 0
        else:
            continue  

    return home_obso, away_obso

def check_event_zone(events, tracking_home, tracking_away):
    # check event zone
    # input:event data format Metrica
    # output:evevnt at attackind third, middle zone, defensive third 
    # zone is based on -52.5~-17.5, -17.5~+17.5, 17.5~52.5
    
    # set zone series format pandas Series
    zone_se = pd.DataFrame(columns=['zone'], index = events.index)   
    # check attack direction
    for event_num in range(len(events)):
        if events.iloc[event_num]['period_id']==1:
            if events.iloc[event_num]['team']=='Home':
                direction = io.BaseDataLoader.find_playing_direction(tracking_home[tracking_home['period_id']==1], 'Home')
            elif events.iloc[event_num]['team']=='Away':
                direction = io.BaseDataLoader.find_playing_direction(tracking_away[tracking_away['period_id']==1], 'Away')
            else:
                direction = 0
        elif events.iloc[event_num]['period_id']==2:
            if events.iloc[event_num]['team']=='Home':
                direction = io.BaseDataLoader.find_playing_direction(tracking_home[tracking_home['period_id']==2], 'Home')
            elif events.iloc[event_num]['team']=='Away':
                direction = io.BaseDataLoader.find_playing_direction(tracking_away[tracking_away['period_id']==2], 'Away')
            else:
                direction = 0

        # add zone defense or middle or attack
        if direction > 0:
            if events.iloc[event_num]['start_x'] < -17.5:
                zone_se.iloc[event_num]['zone'] = 'defense'
            elif events.iloc[event_num]['start_x'] > 17.5:
                zone_se.iloc[event_num]['zone'] = 'attack'
            else:
                zone_se.iloc[event_num]['zone'] = 'middle'
        elif direction < 0:
            if events.iloc[event_num]['start_x'] < -17.5:
                zone_se.iloc[event_num]['zone'] = 'attack'
            elif events.iloc[event_num]['start_x'] > 17.5:
                zone_se.iloc[event_num]['zone'] = 'defense'
            else:
                zone_se.iloc[event_num]['zone'] = 'middle'
        else:
            zone_se.iloc[event_num]['zone'] = 0

    return zone_se

def generate_virtual_trajectory(tracking_data, pass_frame, receive_frame, before_margin_frame, after_margin_frame):
    # generate ghost trajectory
    # input: tracking data (home and away), shot:shot sequence format pandas series
    # output: tracking_home_ghost, tracking_away_ghost
    # set start and end frame

    # simulation margin frame : 1 sec (25 frame)
    start_frame = max(pass_frame - before_margin_frame, receive_frame)
    end_frame = pass_frame + after_margin_frame

    # extract predict all player: 경기 뛰고 있는 모든 선수의 가상의 위치를 예측
    # 단 pass_event['start_frame'] 이전 프레임의 경우 기존 궤적을 그대로 사용하고, 이후 프레임의 경우 속도를 기반으로 직선 궤적을 예측
    tracking_data_ghost = []
    for frame in range(start_frame, pass_frame):
        if frame in tracking_data.index: # skip if frame not in tracking data index: dead state or out of frame range
            tracking_data_ghost.append(tracking_data.loc[frame])
        else:
            pass # skip if frame not in tracking data index
        
    player_cols = [col[:-2] for col in tracking_data.columns if re.match('Home_\d*_x', col) or re.match('Away_\d*_x', col)] + ['ball']
    for frame in range(pass_frame, end_frame+1):
        if frame in tracking_data.index: # skip if frame not in tracking data index: dead state or out of frame range
            new_row = tracking_data.loc[frame].copy()

            for p in player_cols:
                try:
                    # 패스 시점에 패스가 아닌 상황을 가정하여 1프레임 전의 위치/속도를 기반으로 pass_frame 이후 프레임의 위치 예측
                    new_x = tracking_data.at[pass_frame-1, p+'_x'] + tracking_data.at[pass_frame-1, p+'_vx'] * (frame - pass_frame + 1) / 25
                    new_y = tracking_data.at[pass_frame-1, p+'_y'] + tracking_data.at[pass_frame-1, p+'_vy'] * (frame - pass_frame + 1) / 25
                except:
                    # 1프레임 전의 데이터가 없는 경우, 패스 시점의 위치/속도를 기반으로 pass_frame 이후 위치 예측
                    new_x = tracking_data.at[pass_frame, p+'_x'] + tracking_data.at[pass_frame, p+'_vx'] * (frame - pass_frame) / 25
                    new_y = tracking_data.at[pass_frame, p+'_y'] + tracking_data.at[pass_frame, p+'_vy'] * (frame - pass_frame) / 25
                    
                new_row[p+'_x'] = new_x
                new_row[p+'_y'] = new_y
            tracking_data_ghost.append(new_row)
        else:
            pass # skip if frame not in tracking data index

    return pd.DataFrame(tracking_data_ghost)

