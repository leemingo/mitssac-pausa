from logging import config
import re
from elastic.sync import config
import numpy as np
import pandas as pd
import scipy.signal as signal
class ElasticLoader():
    def __init__(self, data_dir, game_id, field_dimen=(105.,68.)):
        self.data_dir = data_dir
        self.game_id = game_id
        self.field_dimen = field_dimen

        self.PASS_EVENTS = ['pass', 'cross']

        self.meta_data = pd.read_parquet(f"{self.data_dir}/{self.game_id}/meta_data.parquet")
        self.events = pd.read_parquet(f"{self.data_dir}/{self.game_id}/event.parquet")
        self.traces = pd.read_parquet(f"{self.data_dir}/{self.game_id}/tracking.parquet") # ball_state = alive
        self.raw_traces = pd.read_parquet(f"{self.data_dir}/{self.game_id}/raw_tracking.parquet")  # ball_state = dead, alive, 

        self.teams = self.get_team_data()
        
        # team ids
        self.home_team_id = str(self.meta_data[self.meta_data['home_away']=='home']['team_id'].values[0])
        self.away_team_id = str(self.meta_data[self.meta_data['home_away']=='away']['team_id'].values[0])

        # player dict: faster mapping using dict
        self.object_id_to_player_id = {d.object_id: d.player_id for d in self.teams.itertuples()}
        self.object_id_to_player_name = {d.object_id: d.player_name for d in self.teams.itertuples()}
        
        # trace_info: faster mapping using dict
        self.trace_info = self.traces.groupby(['period_id', 'frame_id'])[["timestamp", "utc_timestamp", "ball_state", "ball_owning_team_id"]].first().to_dict('index')

    def get_meta_data(self):    
        return self.meta_data
    
    def get_team_data(self):
        teams = self.meta_data.copy()
        teams["team"] = teams["home_away"].apply(
            lambda t: f'{t[0].upper()}{t[1:]}'  # Home, Away
        )
        teams["player_code"] = teams.apply(
            lambda t: f'{t["object_id"][0].upper()}{t["object_id"][1:]}', 
            axis=1
        )
        return teams
        
    def get_event_data(self):
        cols = [
            "game_id",
            "period_id",
            #"utc_timestamp", 
            "frame_id",
            "receive_frame_id",
            'prev_receive_frame_id',
            'team_id',
            'team',
            "player_id",
            'player_name',
            'player_code',
            "receiver_code",
            'spadl_type',
            'success',
            'start_x',
            'start_y',
        ]
        
        events = self.events.copy()
        events = events[
            events["start_x"].notna() &
            events["start_y"].notna() &
            events["frame_id"].notna() & 
            events["player_id"].notna()
        ].reset_index(drop=True)
        events = self.insert_receive_events(events)
        
        events["game_id"] = self.game_id
        events["period_id"] = events["period_id"]

        # mapping player infomation
        events["player_name"] = events["player_id"].map(self.object_id_to_player_name)
        events["player_id"] = events["player_id"].map(self.object_id_to_player_id)
        events["receiver_code"] = events["receiver_id"].apply(lambda id: id.capitalize() if pd.notna(id) else None)
        
        # extract team_id, team, and player_code
        events = events.merge(
            self.teams[['player_id', 'team_id', 'team', 'player_code']],
            how='left',
            on = "player_id",
        )
  
        # convert coordinates
        # x: 0-105 -> -52.5 to 52.5
        # y: 0-68 -> -34 to 34
        for col in ["start_x"]:
            events[col] -= self.field_dimen[0] / 2
        for col in ["start_y"]:
            events[col] -= self.field_dimen[1] / 2

        return events[cols]

    def get_trace_data(self, team_name):
        team = self.teams[self.teams["team"] == team_name].reset_index(drop=True)

        traces = self.traces.copy()
        traces["player_id"] = traces["player_id"].map(self.object_id_to_player_id)
        
        # filter only for players in the team and the ball
        traces = traces[
            (traces.player_id.isin(team.player_id.values)) | 
            (traces.ball) # ball == True
        ].reset_index(drop=True)

        traces = traces.merge(
            team[['player_id', 'player_code']],
            how='left',
            on="player_id"
        )
        traces.loc[traces.ball, 'player_code'] = "ball"
        
        traces = traces.pivot_table(
            index= ['period_id', 'frame_id'],
            columns='player_code',
            values=['x', 'y']
        )

        # player_code: H01, H02, A01, A02, ..., B00
        # value: x, y, vx, vy
        traces.columns = [f'{player_code}_{value}' for value, player_code in traces.columns]    
        traces = traces.reset_index(drop=False)

        # time_seconds: Resets at the start of each half
        # utc_timestamp: UTC timestamp 
        traces["time_seconds"] = traces[["period_id", "frame_id"]].apply(
            lambda row: self.trace_info[(row["period_id"], row["frame_id"])]["timestamp"],
            axis=1
        )
        traces["utc_timestamp"] = traces[["period_id", "frame_id"]].apply(
            lambda row: self.trace_info[(row["period_id"], row["frame_id"])]["utc_timestamp"],
            axis=1
        )
        traces["ball_state"] = traces[["period_id", "frame_id"]].apply(
            lambda row: self.trace_info[(row["period_id"], row["frame_id"])]["ball_state"],
            axis=1
        )
        traces["ball_owning_team_id"] = traces[["period_id", "frame_id"]].apply(
            lambda row: self.trace_info[(row["period_id"], row["frame_id"])]["ball_owning_team_id"],
            axis=1
        )
        traces["team"] = np.where(
            traces["ball_owning_team_id"] == self.home_team_id, "Home",
            np.where(
                traces["ball_owning_team_id"] == self.away_team_id, "Away",
                None
            )
        )
        
        # convert coordinates
        # x: 0-105 -> -52.5 to 52.5
        # y: 0-68 -> -34 to 34
        x_cols = [c for c in traces.columns if re.match(f'{team_name}_\d*_x', c)] + ['ball_x']
        y_cols = [c for c in traces.columns if re.match(f'{team_name}_\d*_y', c)] + ['ball_y']
        traces[x_cols] -= self.field_dimen[0] / 2
        traces[y_cols] -= self.field_dimen[1] / 2   
 
        return traces
    
    def to_single_playing_direction(self, data):
        '''
            Flip coordinates in second half so that each team always shoots in the same direction through the match.
            But, Elastic data already has this applied.
        '''
        
        return data
    
    def convert_common_format_for_event(self, event_df, tracking_home, tracking_away):
        # convert eventdata (from DFL to Metrica)
        # event_df : event data in DFL format

        # set column name
        cols = [
            'game_id',
            'period_id',

            'receive_frame',
            'start_frame',
            'end_frame',
            
            'start_time_seconds',
            'end_time_seconds',

            'player_name',
            'player_code',
            'receiver_code',

            'team',
            'receiver_team',

            'type_name',
            'result_name',

            'start_x',
            'start_y',
            'end_x',
            'end_y'
        ]
        
        common_event_df = pd.DataFrame(columns=cols)
        common_event_df['game_id'] = event_df.game_id
        common_event_df['period_id'] = event_df.period_id
        common_event_df['team'] = event_df.team

        common_event_df['receive_frame'] = event_df.prev_receive_frame_id
        common_event_df['start_frame'] = event_df.frame_id
        common_event_df['end_frame'] = event_df.receive_frame_id
        
        common_event_df['start_time_seconds'] = event_df[['period_id', 'frame_id']].apply(
            lambda row: self.trace_info[(row["period_id"], row["frame_id"])]["timestamp"]
            if pd.notna(row["frame_id"])
            else np.nan,
            axis=1
        )
        common_event_df['end_time_seconds'] = event_df[['period_id', 'receive_frame_id']].apply(
            lambda row: self.trace_info[(row["period_id"], row["receive_frame_id"])]["timestamp"] 
            if pd.notna(row["receive_frame_id"])
            else np.nan,
            axis=1
        )
    
        common_event_df['player_name'] = event_df.player_name
        common_event_df["player_code"] = event_df.player_code
        common_event_df['player_id'] = event_df.player_id
        common_event_df['receiver_code'] = event_df.receiver_code
        common_event_df['receiver_team'] = event_df.receiver_code.apply(lambda id: id.split('_')[0] if pd.notna(id) else None)

        common_event_df['type_name'] = event_df.spadl_type
        common_event_df['result_name'] = event_df.success

        common_event_df['start_x'] = event_df.start_x
        common_event_df['start_y'] = event_df.start_y
        common_event_df['end_x'] = np.nan
        common_event_df['end_y'] = np.nan

        # set end_x, end_y using tracking data
        for (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in zip(common_event_df.groupby("period_id"), tracking_home.groupby("period_id"), tracking_away.groupby("period_id")):
            for idx, row in period_events.iterrows():
                if pd.isna(row.end_frame) or pd.isna(row.receiver_code):
                    continue

                closest_idx = period_tracking_home[period_tracking_home["frame_id"] == row.end_frame].index[0]

                receiver_team = row.receiver_code[0]
                if receiver_team == "H":
                    period_events.at[idx, "end_x"] = period_tracking_home.at[closest_idx, f'{row["receiver_code"]}_x']
                    period_events.at[idx, "end_y"] = period_tracking_home.at[closest_idx, f'{row["receiver_code"]}_y']
                elif receiver_team == "A":
                    period_events.at[idx, "end_x"] = period_tracking_away.at[closest_idx, f'{row["receiver_code"]}_x']
                    period_events.at[idx, "end_y"] = period_tracking_away.at[closest_idx, f'{row["receiver_code"]}_y']
                else:
                    pass

            common_event_df.update(period_events)
            
        return common_event_df[cols]
    
    def insert_receive_events(self, event_df):
        """
            Event data has no 'receive' events, so we insert them here based on the 'pass' events.
        """

        # A Receive event detected by Elastic also include interception or recovery
        # Refer to the function in elastic/sync/receive.py for details.
        pass_like = config.PASS_LIKE_OPEN #+ config.SET_PIECE
        max_frame = 250  # max frame to search for previous receive event (10s at 25fps)

        event_df["sort_priority"] = event_df.index.astype(float)  # for sorting when multiple events occur at the same frame
        receive_event_df = event_df[
            (event_df["receive_frame_id"].notna()) &
            (event_df["receiver_id"].apply(lambda p: p is not None and p[0] in ["h", "a"]) # None / out / goal is excluded
            )
        ].copy()

        if not receive_event_df.empty:
            receive_event_df["sort_priority"] += 0.1  # receive event should come after the pass event at the same frame
            receive_event_df["spadl_type"] = "receive"

            # extract frame_id and player_id
            receive_event_df["utc_timestamp"] = None
            receive_event_df["frame_id"] = receive_event_df["receive_frame_id"]
            receive_event_df["player_id"] = receive_event_df["receiver_id"]
            
            # initialize columns
            receive_event_df[["next_player_id", "next_type", "receiver_id", "receive_frame_id", "synced_ts", "receive_ts"]] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # extract receive position from tracking data.
            position_dict = {(period_id, frame_id, player_id): (x, y) for period_id, frame_id, player_id, x, y in zip(
                self.traces['period_id'],
                self.traces['frame_id'],
                self.traces['player_id'],
                self.traces['x'],
                self.traces['y']
            )}
            receive_event_df[['start_x', 'start_y']] = receive_event_df.apply(
                lambda row: position_dict.get(
                    (row['period_id'], row['frame_id'], row['player_id']),
                    (np.nan, np.nan)
                ),
                axis=1, result_type='expand'
            )
            
            # find previous receive event for each pass-like event
            for receive_row in receive_event_df.itertuples():
                subsequent_events = event_df[
                    (event_df['period_id'] == receive_row.period_id) &
                    (event_df['frame_id'] >= receive_row.frame_id) & 
                    (event_df['frame_id'] <= receive_row.frame_id + max_frame)
                ]

                for subsequent_row in subsequent_events.itertuples():
                    if (
                        (subsequent_row.player_id == receive_row.player_id) and 
                        (subsequent_row.spadl_type in pass_like)
                    ):  
                        event_df.at[subsequent_row.Index, 'prev_receive_frame_id'] = receive_row.frame_id
                        break
                        
            event_df = pd.concat([event_df, receive_event_df], ignore_index=True)
            event_df = event_df.sort_values(
                by=['period_id', 'frame_id', 'sort_priority'], 
                kind='mergesort'
            ).reset_index(drop=True).drop(columns=['sort_priority'])
    
        return event_df
    
    @staticmethod
    def find_playing_direction(team, teamname):
        '''
        Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
        '''    

        GK_column_x = f"{teamname}_{ElasticLoader.find_goalkeeper(team)}_x" # ex) 'H01_x', 'A01_x'
        # +ve is left->right, -ve is right->left
        return -np.sign(team.iloc[0][GK_column_x])
    
    @staticmethod
    def find_goalkeeper(team):
        '''
            Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
            The position of the goalkeeper is at the ends of the field
            where the absolute value is the largest among -52.5 to 52.5.
        ''' 
        x_columns = [c for c in team.columns if re.match('Home_\d*_x', c) or re.match('Away_\d*_x', c)]

        # if the dtype is object due to NaNs, convert to numeric
        first_row = pd.to_numeric(team.iloc[0][x_columns], errors='coerce')
        GK_col = first_row.abs().idxmax(axis=0)
        return GK_col.split('_')[1] # return player id only, ex) '01', '02'

    @staticmethod
    def calc_player_velocities(trace, remove_outliers=True, smoothing=True):
        """ calc_player_velocities( tracking_data )
        
        Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
        https://github.com/hyunsungkim-ds/ballradar/blob/main/datatools/trace_helper.py
        Parameters
        -----------
            team: the tracking DataFrame for home or away team
            smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
            filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
            window: smoothing window size in # of frames
            polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
            maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
            
        Returrns
        -----------
        team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

        """
        DEFAULT_PLAYER_SMOOTHING_PARAMS = {"window_length": 7, "polyorder": 1}
        DEFAULT_BALL_SMOOTHING_PARAMS = {"window_length": 3, "polyorder": 1}
        MAX_PLAYER_SPEED: float = 12.0
        MAX_BALL_SPEED: float = 28.0

        # remove any velocity data already in the dataframe
        columns = [c for c in trace.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
        trace = trace.drop(columns=columns)

        # Get the player ids
        player_ids = np.unique([c[:-2] for c in trace.columns if c[:4] in ['Home','Away']])
        player_ids = list(player_ids) + ['ball'] # include ball

        # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half    
        dt = trace['time_seconds'].diff()
        
        # index of first frame in second half
        second_half_idx = trace.period_id.idxmax()
        
        # estimate velocities for players in team
        for player in player_ids: # cycle through players individually
            smoothing_params = DEFAULT_BALL_SMOOTHING_PARAMS if player == 'ball' else DEFAULT_PLAYER_SMOOTHING_PARAMS
            max_speed = MAX_BALL_SPEED if player == 'ball' else MAX_PLAYER_SPEED

            x = trace[player+"_x"].copy()
            y = trace[player+"_y"].copy()

            # position smoothing
            # if smoothing:
            #     x.loc[:second_half_idx] = signal.savgol_filter(x.loc[:second_half_idx], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])
            #     y.loc[:second_half_idx] = signal.savgol_filter(y.loc[:second_half_idx], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])

            #     x.loc[second_half_idx:] = signal.savgol_filter(x.loc[second_half_idx:], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])
            #     y.loc[second_half_idx:] = signal.savgol_filter(y.loc[second_half_idx:], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])

            # difference player positions in timestep dt to get unsmoothed estimate of velicity
            vx = pd.Series(index=x.index, dtype=float)
            vy = pd.Series(index=y.index, dtype=float)
            vx.loc[:second_half_idx] = x.loc[:second_half_idx].diff() / dt.loc[:second_half_idx]
            vy.loc[:second_half_idx] = y.loc[:second_half_idx].diff() / dt.loc[:second_half_idx]

            vx.loc[second_half_idx:] = x.loc[second_half_idx:].diff() / dt.loc[second_half_idx:]
            vy.loc[second_half_idx:] = y.loc[second_half_idx:].diff() / dt.loc[second_half_idx:]

            if remove_outliers:
                speeds = np.sqrt(vx.loc[:second_half_idx]**2 + vy.loc[:second_half_idx]**2)
                is_speed_outlier = speeds > max_speed
                vx.loc[:second_half_idx] = pd.Series(np.where(is_speed_outlier, np.nan, vx.loc[:second_half_idx])).interpolate(limit_direction="both").values
                vy.loc[:second_half_idx] = pd.Series(np.where(is_speed_outlier, np.nan, vy.loc[:second_half_idx])).interpolate(limit_direction="both").values

                speeds = np.sqrt(vx.loc[second_half_idx:]**2 + vy.loc[second_half_idx:]**2)
                is_speed_outlier = speeds > max_speed
                vx.loc[second_half_idx:] = pd.Series(np.where(is_speed_outlier, np.nan, vx.loc[second_half_idx:])).interpolate(limit_direction="both").values   
                vy.loc[second_half_idx:] = pd.Series(np.where(is_speed_outlier, np.nan, vy.loc[second_half_idx:])).interpolate(limit_direction="both").values

            # smooth velocities
            if smoothing:
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"])

                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"]) 
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:], window_length=smoothing_params["window_length"], polyorder=smoothing_params["polyorder"]) 

            trace[player + "_vx"] = vx
            trace[player + "_vy"] = vy
            #trace[player + "_speed"] = np.sqrt(vx**2 + vy**2)

        return trace
    
    @staticmethod
    def check_home_away_event(events, tracking_home, tracking_away):
        # check wether corresponded event data and tracking data defined as 'Home' or 'Away'
        # input : events in format Metrica, tracking data in format Metrica
        
        # search nearest player home and away
        # set player name (ex. Home_1, ...)
        home_name = [p[:-2] for p in tracking_home.columns if re.match('Home_\d*_x', p)]
        away_name = [p[:-2] for p in tracking_away.columns if re.match('Away_\d*_x', p)]

        # calculate distace player to ball
        # set home distance
        home_dis = []
        for player in home_name:
            # Exception handling for no entry player
            if np.isnan(tracking_home[player+'_x'].iloc[0]):
                continue
            else:
                ball_pos = np.array([tracking_home['ball_x'].iloc[0], tracking_home['ball_y'].iloc[0]])
                player_pos = np.array([tracking_home[player+'_x'].iloc[0], tracking_home[player+'_y'].iloc[0]])
                home_dis.append(np.linalg.norm(player_pos - ball_pos))

        # set away distance
        away_dis = []
        for player in away_name:
            # Exception handling for no entry player
            if np.isnan(tracking_away[player+'_x'].iloc[0]):
                continue
            else:
                ball_pos = np.array([tracking_away['ball_x'].iloc[0], tracking_away['ball_y'].iloc[0]])
                player_pos = np.array([tracking_away[player+'_x'].iloc[0], tracking_away[player+'_y'].iloc[0]])
                away_dis.append(np.linalg.norm(player_pos - ball_pos))

        # judge kick-off team
        if min(home_dis) < min(away_dis):
            kickoff_team = 'Home'
        else:
            kickoff_team = 'Away'
        
        # check team in events
        if events.iloc[0].team != 'Home' and events.iloc[0].team != 'Away':
            raise ValueError("Team name in event data is not defined as 'Home' or 'Away'")
        
        if kickoff_team != events.iloc[0].team:
            # replace 'Home' to 'Away' and 'Away' to 'Home'
            events = events.replace({'team':{'Home':'Away', 'Away':'Home'}})
            print(f'Change Team Name: kickoff={kickoff_team}, event first team= {events.iloc[0].team}')
        else:
            print(f'Team Name OK: kickoff={kickoff_team}, event first team= {events.iloc[0].team}')
    
        return events