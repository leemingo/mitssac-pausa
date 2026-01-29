import os
import json
import xml.etree.ElementTree as ET
import fnmatch
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from kloppy import sportec
from kloppy.domain import Dimension, MetricPitchDimensions, Orientation, TrackingDataset

from sync.config import PITCH_X, PITCH_Y
from tools.match_data import MatchData
from elastic.tools import bepro_spadl

pd.set_option('future.no_silent_downcasting', True)

class BeproData(MatchData):
    def __init__(self, root_dir: str, match_id: str, load_tracking: bool = True):
        """
        Bepro data loader and processor.
        Args:   
            root_dir: Root directory containing match data.
            match_id: Match identifier.
            load_tracking: Whether to load tracking data immediately.
            load: Whether to load all data upon initialization.
        """
        
        super().__init__()
        self.match_id = match_id
        match_path = os.path.join(root_dir, match_id)

        meta_files = [f for f in os.listdir(match_path) if "metadata" in f]
        # event_files = sorted([f for f in os.listdir(match_path) if "1st Half" in f or "2nd Half" in f])
        event_files = sorted([f for f in os.listdir(match_path) if "event" in f])
        tracking_files = sorted([f for f in os.listdir(match_path) if "1_frame_data" in f or "2_frame_data" in f])
        assert meta_files and event_files and tracking_files, f"Required files are missing in {match_path}"

        self.meta_path = f"{match_path}/{meta_files[0]}"
        self.event_path = [f"{match_path}/{event_files[0]}", f"{match_path}/{event_files[1]}"]
        self.tracking_path = [f"{match_path}/{tracking_files[0]}", f"{match_path}/{tracking_files[1]}"]
        
        self.meta_data = self.load_meta_data(self.meta_path)
        self.fps, self.ground_width, self.ground_height = self.meta_data["fps"], self.meta_data["ground_width"], self.meta_data["ground_height"]
        self.fps = 25 # resample to 25hz
        self.lineup = self.load_lineup_data(self.meta_path)
        self.events = self.load_event_data(self.event_path)
        self.events = self.align_event_identifier(self.lineup, self.events, self.match_id)
        self.events = self.align_event_orientations(self.lineup, self.events)

        # Since it often takes more than a minute to load tracking data, you can choose whether to delay loading
        if load_tracking:
            self.tracking = self.load_tracking_data(self.tracking_path, self.meta_data, self.lineup)
            self.tracking = self.align_trace_orientations(self.lineup, self.tracking)
     
    @staticmethod
    def load_meta_data(meta_path: str) -> dict:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        return meta_data
    
    @staticmethod
    def load_lineup_data(meta_path: str) -> pd.DataFrame:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
 
        home_team_info, away_team_info = meta_data['home_team'], meta_data['away_team']
        home_team_rows = []

        base_info = {
                'team_id': home_team_info.get('team_id'),
                'team_name': home_team_info.get('team_name'),
                'home_away': 'home',
                'player_id': None, 
                'uniform_number': None,
                'object_id': None,
                'player_name': None,
                'playing_position': None,
            }
        for _, player_data in enumerate(home_team_info['players']):
            player_info = base_info.copy()
            player_info['player_name'] = player_data['full_name_en']
            player_info['playing_position'] = player_data['initial_position_name']
            player_info['uniform_number'] = player_data['shirt_number']
            player_info['player_id'] = player_data['player_id']
            player_info['object_id'] = f"home_{player_data['shirt_number']}"

            home_team_rows.append(player_info)

        home_df = pd.DataFrame(home_team_rows)
        home_df['player_id'] = home_df['player_id'].astype(int)
        home_df['team_id'] = home_df['team_id'].astype(int)

        away_team_rows = []
        base_info['home_away'] = 'away'
        base_info['team_id'] = away_team_info.get('team_id')
        base_info['team_name'] = away_team_info.get('team_name')
        for _, player_data in enumerate(away_team_info['players']):
            player_info = base_info.copy()
            player_info['player_name'] = player_data['full_name_en']
            player_info['playing_position'] = player_data['initial_position_name']
            player_info['uniform_number'] = player_data['shirt_number']
            player_info['player_id'] = player_data['player_id']
            player_info['object_id'] = f"away_{player_data['shirt_number']}"

            away_team_rows.append(player_info)

        away_df = pd.DataFrame(away_team_rows)
        away_df['player_id'] = away_df['player_id'].astype(int)
        away_df['team_id'] = away_df['team_id'].astype(int)

        return pd.concat([home_df, away_df], ignore_index=True)

    @staticmethod
    def load_event_data(event_path: str) -> pd.DataFrame:
        with open(event_path[0], 'r', encoding='utf-8') as f:
            first_half_event_data = json.load(f)
        with open(event_path[1], 'r', encoding='utf-8') as f:
            second_half_event_data = json.load(f)

        first_half_event_df = pd.DataFrame(first_half_event_data['data'])
        second_half_event_df = pd.DataFrame(second_half_event_data['data'])
        events = pd.concat([first_half_event_df, second_half_event_df], axis=0, ignore_index=True)

        events["period_id"] = events["period_order"] + 1 
        for col in ["x", "to_x"]:
            events[col] = events[col] * PITCH_X
        for col in ["y", "to_y"]:
            events[col] = events[col] * PITCH_Y
        
        return events

    @staticmethod
    def load_tracking_data(tracking_path: str, meta_data: pd.DataFrame, lineup: pd.DataFrame) -> Tuple[TrackingDataset, pd.DataFrame]:
        player_lookup = lineup.set_index('player_id')
        home_tid = lineup[lineup['home_away'] == 'home']['team_id'].iloc[0]
        away_tid = lineup[lineup['home_away'] == 'away']['team_id'].iloc[0]
        
        # Configuration
        player_smoothing_params = {"window_length": 7, "polyorder": 1}
        ball_smoothing_params = {"window_length": 3, "polyorder": 1}
        max_player_speed = 12.0
        max_player_acceleration = 6.0
        max_ball_speed = 28.0
        max_ball_acceleration = 13.5

        first_half_tracking_data = []
        with open(tracking_path[0], 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                processed_line = line.strip()
                if not processed_line:
                    continue
                first_half_tracking_data.append(json.loads(processed_line))

        second_half_tracking_data = []
        with open(tracking_path[1], 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                processed_line = line.strip()
                if not processed_line:
                    continue
                second_half_tracking_data.append(json.loads(processed_line))

        all_object_rows = []
        for half_tracking_data in [first_half_tracking_data, second_half_tracking_data]:
            for frame_data in half_tracking_data:
                # Check ball state
                ball_state = frame_data.get('ball_state')
                if ball_state is None or ball_state == 'out':
                    new_ball_state = 'dead'
                    ball_owning_team_id = None
                else:
                    new_ball_state = 'alive'
                    ball_owning_team_id = home_tid if ball_state == 'home' else (away_tid if ball_state == 'away' else ball_state)
                    
                # Extract frame information
                frame_info = {
                    'game_id': meta_data.get('match_id'),
                    'period_id': frame_data.get('period_order') + 1,
                    'timestamp': frame_data.get('match_time'),
                    'frame_id': frame_data.get('frame_index'),
                    'ball_state': new_ball_state,
                    'ball_owning_team_id': ball_owning_team_id,
                }

                for object_type in ['players', 'balls']:
                    object_list = frame_data.get(object_type, [])
                    if object_list:
                        for object_data in object_list:
                            row_data = frame_info.copy()
                            row_data.update(object_data)
                            
                            if object_type == 'balls':
                                row_data.update({
                                    'id': 'ball',
                                    'team_id': 'ball',
                                    'position_name': 'ball',
                                    'object_id': 'ball',
                                })
                            else:
                                player_pID = int(object_data.get('player_id'))
                                row_data['id'] = player_pID
                                if player_pID in player_lookup.index:
                                    row_data['object_id'] = player_lookup.loc[player_pID, 'object_id']
                                    row_data['team_id'] = player_lookup.loc[player_pID, 'team_id']
                                    row_data['position_name'] = player_lookup.loc[player_pID, 'playing_position']
                                else:
                                    raise ValueError(f"Player ID {player_pID} not found in lineup data.\n{player_lookup}")
                            
                            # Remove unnecessary columns
                            row_data.pop('object', None)
                            row_data.pop('player_id', None)
                            all_object_rows.append(row_data)

        tracking_df = pd.DataFrame(all_object_rows)
        tracking_df['timestamp'] = pd.to_timedelta(tracking_df['timestamp'], unit='ms')

        # Rescale pitch coordinates
        tracking_df = BeproData.rescale_pitch(tracking_df, meta_data)
        tracking_df = BeproData.resample_tracking_dataframe(tracking_df, target_hz=25)
        
        # Calculate kinematics for each agent
        total_tracking_list = []
        for agent_id in tracking_df['id'].unique():
            is_ball = (agent_id == 'ball')
            current_agent_df = tracking_df[tracking_df['id'] == agent_id].copy()
            current_agent_df['z'] = 0.0  # bepro data doesn't have z information

            # Drop rows with NaN coordinates for players
            if not is_ball:
                current_agent_df = current_agent_df.dropna(subset=['x', 'y']).copy()

            # Calculate kinematics
            smoothing = ball_smoothing_params if is_ball else player_smoothing_params
            max_v = max_ball_speed if is_ball else max_player_speed
            max_a = max_ball_acceleration if is_ball else max_player_acceleration

            kinematics_df = BeproData._calculate_kinematics(current_agent_df, smoothing, max_v, max_a, is_ball)
            total_tracking_list.append(kinematics_df)
        
        total_tracking_df = pd.concat(total_tracking_list, axis=0, ignore_index=True)
        
        # Sort and format final DataFrame
        total_tracking_df = total_tracking_df.sort_values(
            by=["period_id", "timestamp", "frame_id", "id"], kind="mergesort"
        ).reset_index(drop=True)

        # Define final column order
        final_cols_order = [
            'game_id', 'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'v', 'ax', 'ay', 'az', 'a',
            'id', 'object_id', 'team_id', 'position_name'
        ]
        total_tracking_df = total_tracking_df[final_cols_order]

        # Convert datatypes
        total_tracking_df['ball_owning_team_id'] = total_tracking_df['ball_owning_team_id'].astype(str)

        # determine ball owning team at each frame
        #ball_owning_team_id: bepro_precessor.py 참고
        g = total_tracking_df.groupby(['period_id', 'timestamp', 'frame_id'])
        ball_state = pd.DataFrame({
            "ball_state": g["ball_state"].apply(lambda s: s.value_counts().idxmax()),
            "ball_owning_team_id": g["ball_owning_team_id"].apply(lambda s: s.value_counts().idxmax()),
        }).reset_index(drop=False) # index (period_id, timestamp, frame_id) to columns

        total_tracking_df = total_tracking_df.pivot_table(
            index= ['period_id', 'timestamp', 'frame_id'],
            columns='object_id',
            values=['x', 'y', 'z']#, 'vx', 'vy']
        )

        # value: x, y, vx, vy
        # player_code: H01, H02, A01, A02, ..., B00
        total_tracking_df.columns = [f'{player_code}_{value}' for value, player_code in total_tracking_df.columns]    
        total_tracking_df = total_tracking_df.reset_index(drop=False)

        total_tracking_df = total_tracking_df.merge(
            ball_state,
            how='left',
            on=['period_id', 'timestamp', 'frame_id']
        )
        total_tracking_df.loc[total_tracking_df['ball_x'].isna() | total_tracking_df['ball_y'].isna(), 'ball_state'] = 'dead'
        total_tracking_df.loc[total_tracking_df['ball_x'].isna() | total_tracking_df['ball_y'].isna(), 'ball_owning_team_id'] = None
    
        # timestamp: milliseconds) -> timestamp(seconds): 초 단위로 변횐
        total_tracking_df['timestamp'] = (
            total_tracking_df['timestamp'].dt.total_seconds()
            - ((total_tracking_df.period_id > 1) * 45 * 60)
            - ((total_tracking_df.period_id > 2) * 15 * 60)
            - ((total_tracking_df.period_id > 3) * 15 * 60)
        )

        return total_tracking_df

    @staticmethod
    def _calculate_kinematics(df: pd.DataFrame, smoothing_params: Dict, max_speed: float, 
                            max_acceleration: float, is_ball: bool = False) -> pd.DataFrame:
        """Calculates velocity and acceleration for a single agent over periods.
        
        This function implements a comprehensive kinematics calculation pipeline for
        tracking data. It processes data period by period, calculating velocities and
        accelerations while applying outlier detection and smoothing techniques.
        
        Args:
            df: DataFrame containing tracking data with columns ['x', 'y', 'z', 'timestamp', 'period_id'].
            smoothing_params: Dictionary containing smoothing parameters for Savitzky-Golay filter.
            max_speed: Maximum allowed speed value for outlier detection.
            max_acceleration: Maximum allowed acceleration value for outlier detection.
            is_ball: Boolean indicating if the agent is a ball (affects z-coordinate handling).
            
        Returns:
            DataFrame with calculated kinematics including velocity (vx, vy, vz, v) and 
            acceleration (ax, ay, az, a) columns.
            
        Example:
            >>> kinematics_df = _calculate_kinematics(
            ...     tracking_df, smoothing_params, max_speed=12.0, max_acceleration=15.0, is_ball=False
            ... )
        """
        def _apply_smoothing_and_outlier_removal(period_df: pd.DataFrame, col: str, 
                                            is_outlier: pd.Series, smoothing_params: Dict) -> pd.DataFrame:
            """Helper function to apply smoothing and outlier removal to a column.
            
            This function implements a comprehensive smoothing and outlier removal pipeline
            using Savitzky-Golay filtering. It first masks outliers, interpolates missing
            values, and then applies smoothing with appropriate parameter validation.
            
            Args:
                period_df: DataFrame containing the period data.
                col: Column name to apply smoothing to.
                is_outlier: Boolean Series indicating outlier values.
                smoothing_params: Dictionary containing smoothing parameters including
                                'window_length' and 'polyorder'.
                
            Returns:
                DataFrame with smoothed column values.
                
            Example:
                >>> smoothed_df = _apply_smoothing_and_outlier_removal(
                ...     period_df, 'vx', is_outlier, {'window_length': 11, 'polyorder': 3}
                ... )
            """
            period_df[col] = period_df[col].mask(is_outlier)
            period_df[col] = period_df[col].interpolate(limit_direction='both')
            
            data_to_smooth = period_df[col].fillna(0)
            window_length = min(smoothing_params['window_length'], len(data_to_smooth))
            if window_length % 2 == 0:
                window_length -= 1
            
            if window_length >= smoothing_params['polyorder'] + 1 and window_length > 0:
                period_df[col] = savgol_filter(data_to_smooth, window_length=window_length, polyorder=smoothing_params['polyorder'])
            else:
                period_df[col] = data_to_smooth
            
            return period_df


        df_out = pd.DataFrame()
        required_cols = ['x', 'y', 'z', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in input dataframe. Found: {df.columns.tolist()}")
            return df_out

        for period_id in df['period_id'].unique():
            period_df = df[df['period_id'] == period_id].copy()
            period_df = period_df.sort_values(by='timestamp').reset_index(drop=True)

            # Interpolate coordinates
            period_df['x'] = period_df['x'].interpolate()
            period_df['y'] = period_df['y'].interpolate()

            # Calculate time difference
            dt = period_df['timestamp'].diff().dt.total_seconds()

            # Calculate velocities
            coord_cols = ['x', 'y', 'z']
            vel_cols = ['vx', 'vy', 'vz']
            accel_cols = ['ax', 'ay', 'az']
            
            for vel_col, coord_col in zip(vel_cols, coord_cols):
                period_df[vel_col] = period_df[coord_col].diff() / dt
                if vel_col == 'vz':
                    period_df[vel_col] = 0.0

            # Calculate speed and apply outlier removal
            period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
            is_speed_outlier = period_df['v'] > max_speed
            
            for col in vel_cols:
                period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_speed_outlier, smoothing_params)
            
            # Recalculate speed after smoothing
            period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
            
            # Calculate accelerations
            for accel_col, vel_col in zip(accel_cols, vel_cols):
                period_df[accel_col] = period_df[vel_col].diff() / dt
                if accel_col == 'az':
                    period_df[accel_col] = 0.0

            # Calculate acceleration magnitude and apply outlier removal
            period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
            is_accel_outlier = period_df['a'] > max_acceleration
            
            for col in accel_cols:
                period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_accel_outlier, smoothing_params)
            
            # Recalculate acceleration after smoothing
            period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
            
            # Limit speed and acceleration
            period_df['v'] = np.minimum(period_df['v'], max_speed)
            period_df['a'] = np.minimum(period_df['a'], max_acceleration)

            df_out = pd.concat([df_out, period_df], ignore_index=True)

        return df_out

    @staticmethod
    def rescale_pitch(tracking_df: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
        """Rescales pitch coordinates to standard dimensions.
        
        This function transforms pitch coordinates from the original coordinate system
        to a standardized pitch coordinate system. It handles both x and y coordinates
        and applies appropriate scaling factors based on the pitch metadata.
        
        Args:
            tracking_df: DataFrame containing tracking data with x, y coordinates.
            meta_data: Dictionary containing pitch metadata including ground_width and ground_height.
            
        Returns:
            DataFrame with rescaled x, y coordinates to standard pitch dimensions.
            
        Example:
            >>> rescaled_df = rescale_pitch(tracking_df, meta_data)
        """
        x_ori_min, x_ori_max = 0.0, meta_data['ground_width']
        y_ori_min, y_ori_max = 0.0, meta_data['ground_height']

        x_new_min, x_new_max = 0, PITCH_X
        y_new_min, y_new_max = 0, PITCH_Y

        scale_x = (x_new_max - x_new_min) / (x_ori_max - x_ori_min)
        scale_y = (y_new_max - y_new_min) / (y_ori_max - y_ori_min)

        tracking_df['x'] = x_new_min + (tracking_df['x'] - x_ori_min) * scale_x
        tracking_df['y'] = y_new_min + (tracking_df['y'] - y_ori_min) * scale_y
        
        return tracking_df

    @staticmethod
    def resample_tracking_dataframe(tracking_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
        """Resamples tracking data to target frequency.
        
        This function resamples tracking data from its original frequency to a target
        frequency using interpolation techniques. It handles both forward and backward
        filling for different types of data columns and ensures proper time alignment.
        
        Args:
            tracking_df: DataFrame containing tracking data with timestamp index.
            target_hz: Target frequency in Hz for resampling.
            
        Returns:
            Resampled DataFrame with data at the target frequency.
            
        Example:
            >>> resampled_df = resample_tracking_dataframe(tracking_df, target_hz=25)
        """
        resample_freq_ms = int(1000 / target_hz)
        resample_freq_str = f'{resample_freq_ms}ms'
        
        period_list = []
        for period_id in tracking_df['period_id'].unique():
            period_df = tracking_df[tracking_df['period_id'] == period_id]

            min_timestamp = pd.Timedelta(0)
            max_timestamp = period_df['timestamp'].max()
            global_original_index = pd.to_timedelta(sorted(period_df['timestamp'].unique()))
            global_target_index = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=resample_freq_str)

            grouped = period_df.groupby('id')
            resampled_list = []

            for agent_id, agent_group in grouped:
                group_df = agent_group.copy().set_index('timestamp')
                if group_df.index.has_duplicates:
                    group_df = group_df.loc[~group_df.index.duplicated(keep='first')]

                union_index = global_original_index.union(global_target_index)
                reindexed_group = group_df.reindex(union_index)

                # Interpolation
                interpolation_cols = ['x', 'y', 'speed']
                reindexed_group[interpolation_cols] = reindexed_group[interpolation_cols].interpolate(method='pchip', limit_area='inside')
                
                # Forward fill other columns
                ffill_cols = [col for col in group_df.columns if col not in interpolation_cols and col != 'id']
                reindexed_group[ffill_cols] = reindexed_group[ffill_cols].ffill()
                final_group = reindexed_group.reindex(global_target_index)

                # Fill categorical data
                final_group['id'] = agent_id
                final_group = final_group.dropna(subset=['x', 'y'])
                resampled_list.append(final_group)

            period_list += resampled_list

        total_resampled_df = pd.concat(period_list).reset_index().rename(columns={'index': 'timestamp'})
        total_resampled_df['frame_id'] = (total_resampled_df['timestamp'].astype(np.int64) // (10**9 / target_hz)).astype(int)
        total_resampled_df = total_resampled_df.sort_values(['timestamp', 'period_id', 'frame_id', 'id'])

        return total_resampled_df

    @staticmethod
    def align_event_identifier(lineup: pd.DataFrame, events: pd.DataFrame, match_id: int) -> pd.DataFrame:
        if "player_id" not in events.columns:
            player_name_to_id = lineup.set_index("player_name")["player_id"].to_dict()
            events["player_id"] = events["player_name"].map(player_name_to_id)
        
        if "team_id" not in events.columns:
            player_id_to_team_id = lineup.set_index("player_id")["team_id"].to_dict()
            events["team_id"] = events["player_id"].map(player_id_to_team_id)
        
        if "match_id" not in events.columns:
            events["match_id"] = match_id

        if "event_id" not in events.columns:
            events["event_id"] = events.apply(lambda ets: f"{ets.match_id}_{ets.period_id}_{ets.name}", axis=1)
        
        return events

    @staticmethod
    def align_event_orientations(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
            Rotate events so that the home team plays on the left side
        """
        events = events.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "home", "player_id"].tolist()
        away_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "away", "player_id"].tolist()

        for period_id in events["period_id"].unique():
            period_events = events[events["period_id"] == period_id].copy()
            home_gk_x = period_events.loc[period_events["player_id"].isin(home_gk_ids), "x"]
            away_gk_x = period_events.loc[period_events["player_id"].isin(away_gk_ids), "x"]

            if home_gk_x.mean() > away_gk_x.mean():  
                events.loc[period_events.index, "x"] = (PITCH_X - period_events["x"]).round(2)
                events.loc[period_events.index, "to_x"] = (PITCH_X - period_events["to_x"]).round(2)
                events.loc[period_events.index, "y"] = (PITCH_Y - period_events["y"]).round(2)
                events.loc[period_events.index, "to_y"] = (PITCH_Y - period_events["to_y"]).round(2)

        return events

    @staticmethod
    def align_trace_orientations(lineup: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
        """
            Rotate traces so that the home team plays on the left side
        """

        traces = traces.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "home", "object_id"].tolist()
        away_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "away", "object_id"].tolist()

        for period_id in traces["period_id"].unique():
            period_traces = traces[traces["period_id"] == period_id].copy()
            home_gk_x = period_traces[[f"{p}_x" for p in home_gk_ids]].values
            away_gk_x = period_traces[[f"{p}_x" for p in away_gk_ids]].values
            
            x_cols = [col for col in traces.columns if col.endswith("_x")]
            y_cols = [col for col in traces.columns if col.endswith("_y")]

            if home_gk_x.mean() > away_gk_x.mean():
                traces.loc[period_traces.index, x_cols] = PITCH_X - traces.loc[period_traces.index, x_cols].values
                traces.loc[period_traces.index, y_cols] = PITCH_Y - traces.loc[period_traces.index, y_cols].values

        return traces

    @staticmethod
    def find_object_ids(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        events["object_id"] = events["player_id"].map(player_mapping)
        # bepro: not exist receiver_player_id
        events["receiver_id"] = np.nan # events["receiver_player_id"].map(player_mapping)

        return events

    @staticmethod
    def find_spadl_event_types(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events = bepro_spadl.convert_to_actions(events)

        return events

    def format_events_for_syncer(self) -> pd.DataFrame:
        events = self.find_object_ids(self.lineup, self.events)
        events = self.find_spadl_event_types(events)
      
        events["utc_timestamp"] = (
            pd.to_datetime(self.meta_data.get("match_datetime")).tz_convert("UTC").tz_localize(None) + 
            pd.to_timedelta(events["time_seconds"], unit='s') +
            ((events["period_id"] > 1) * pd.to_timedelta(45, unit='m')) +
            ((events["period_id"] > 2) * pd.to_timedelta(15, unit='m')) +
            ((events["period_id"] > 3) * pd.to_timedelta(15, unit='m'))
        )
        column_mapping = {
            "period_id": "period_id",
            "utc_timestamp": "utc_timestamp",
            "object_id": "player_id",
            "type_name": "spadl_type",
            "start_x": "start_x",
            "start_y": "start_y",
            "success": "success",
        }
        input_events = events.loc[events["type_name"].notna(), column_mapping.keys()].copy().reset_index(drop=True)
        input_events = input_events.rename(columns=column_mapping).astype({"success": bool})
        input_events = input_events[input_events["player_id"].notna()].reset_index(drop=True)
        input_events = input_events[input_events["spadl_type"] != "dribble"].reset_index(drop=True)

        return input_events
    
    # 인근 시점이 alive 상태가 아닌 이벤트 제거
    def get_alive_events(self, input_events: pd.DataFrame, input_tracking: pd.DataFrame) -> List[int]:
        alive_event_indices = []

        for idx, row in input_events.iterrows():
            tracking_period = input_tracking[input_tracking["period_id"] == row.period_id]
            time_window_start = row.utc_timestamp - pd.Timedelta(seconds=1)
            time_window_end = row.utc_timestamp + pd.Timedelta(seconds=1)

            tracking_window = tracking_period[
                (tracking_period["utc_timestamp"] >= time_window_start) & 
                (tracking_period["utc_timestamp"] <= time_window_end)
            ]

            if not tracking_window.empty and (tracking_window["ball_state"] == "alive").any():
                alive_event_indices.append(idx)
        
        input_events = input_events.loc[alive_event_indices].reset_index(drop=True)
        return input_events
    
    def format_tracking_for_syncer(self) -> pd.DataFrame:
        tracking = self.tracking.copy()
        tracking = tracking.drop(columns=["frame_id"]) # elastic 형태에 맞게 재변형

        if "frame_id" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            # Bepro use a fixed tracking timestamp that does not start at kickoff.
            # In such cases, preserve the defined timestamp (e.g. match_datetime).
            tracking = MatchData.calculate_tracking_datetimes(events=None, tracking=tracking, fps=self.fps)
            tracking["utc_timestamp"] = (
                pd.to_datetime(self.meta_data.get("match_datetime")).tz_convert("UTC").tz_localize(None) + 
                pd.to_timedelta(tracking["timestamp"], unit='s') +
                ((tracking["period_id"] > 1) * pd.to_timedelta(45, unit='m')) +
                ((tracking["period_id"] > 2) * pd.to_timedelta(15, unit='m')) +
                ((tracking["period_id"] > 3) * pd.to_timedelta(15, unit='m'))
            )

        home_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]
        objects = home_players + away_players + ["ball"]
        
        tracking_list = []
        for p in objects:
            object_tracking = tracking[["frame_id", "period_id", "timestamp", "utc_timestamp", "ball_state", "ball_owning_team_id"]].copy()

            if p == "ball":
                object_tracking["player_id"] = None
                object_tracking["ball"] = True
            else:
                object_tracking["player_id"] = p
                object_tracking["ball"] = False

            object_tracking["x"] = tracking[f"{p}_x"].values.round(2)
            object_tracking["y"] = tracking[f"{p}_y"].values.round(2)
            object_tracking["z"] = tracking["ball_z"].values.round(2) if p == "ball" else np.nan

            for i in object_tracking["period_id"].unique():                
                period_tracking = object_tracking[object_tracking["period_id"] == i].dropna(subset=["x"]).copy()
                if not period_tracking.empty:
                    vx = savgol_filter(np.diff(period_tracking["x"].values) * self.fps, window_length=15, polyorder=2)
                    vy = savgol_filter(np.diff(period_tracking["y"].values) * self.fps, window_length=15, polyorder=2)
                    speed = np.sqrt(vx**2 + vy**2)
                    period_tracking.loc[period_tracking.index[1:], "speed"] = speed
                    period_tracking["speed"] = period_tracking["speed"].bfill()

                    accel = savgol_filter(np.diff(speed) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_s"] = accel
                    period_tracking["accel_s"] = period_tracking["accel_s"].bfill().ffill()

                    ax = savgol_filter(np.diff(vx) * self.fps, window_length=9, polyorder=2)
                    ay = savgol_filter(np.diff(vy) * self.fps, window_length=9, polyorder=2)
                    period_tracking.loc[period_tracking.index[1:-1], "accel_v"] = np.sqrt(ax**2 + ay**2)
                    period_tracking["accel_v"] = period_tracking["accel_v"].bfill().ffill()
                    tracking_list.append(period_tracking)
                else:
                    if p == "ball":
                        print(f"Warning: No tracking data for period {i} of object {p}.")
                        print(period_tracking)
    
        tracking_data = pd.concat(tracking_list, ignore_index=True)
        
        raw_tracking = tracking_data.reset_index(drop=True).astype({"period_id": int, "z": float})
        input_tracking = tracking_data[tracking_data["ball_state"] == "alive"].reset_index(drop=True).astype({"period_id": int, "z": float})
        
        return raw_tracking, input_tracking