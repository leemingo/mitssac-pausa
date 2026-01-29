import argparse
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

import loader as io
import pitch_control as pc
import obso as obs

warnings.simplefilter('ignore')

FIELD_DIMS = (105, 68)  
BEFORE_MARGIN_FRAME = 75 # 75fps = 3seconds
AFTER_MARGIN_FRAME = 25  # 25fps = 1second
PASS_EVENTS = ['pass', 'cross']
RECEIVE_EVENTS = ['receive'] # only receive event. why? A Receive event detected by Elastic also include interception or recovery.

def get_loader(provider, data_dir, game_id):
    """데이터 제공자에 따라 적절한 로더를 반환합니다."""
    loader = None
    if provider == 'bepro':
        loader = io.ElasticLoader(data_dir, game_id)
    elif provider == 'dfl':
        loader = io.ElasticLoader(data_dir, game_id)
    else:
        raise ValueError(f"Unsupported data provider: {provider}")
    
    return loader

def save_obso(results: dict, output_dir: str):
    """
        save OBSO results to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}...")
    for key, df in results.items():
        file_path = os.path.join(output_dir, f"{key}.pkl")

        with open(file_path, 'wb') as file:
            pickle.dump(df, file)
    
# Saving OBSO results with compressed map data
def save_obso_compressed(results: dict, output_dir: str):
    """ 
        save OBSO results with compressed map data to reduce storage space.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}...")
    
    HEAVY_COLS = ['obso_map', 'ppcf_map', 'trans_map', 'score_map']
    for key, df in results.items():    
        file_path = os.path.join(output_dir, f"{key}.pkl")     # save meta data without heavy map columns (.pkl)
        npz_path = os.path.join(output_dir, f"{key}_maps.npz") # save heavy map data separately (.npz)
                    
        existing_heavy_cols = [col for col in HEAVY_COLS if col in df.columns]
        if existing_heavy_cols:
            np.savez_compressed(npz_path, **{col: np.stack(df[col].values) for col in existing_heavy_cols})

        with open(file_path, 'wb') as file:
            pickle.dump(df.drop(columns=existing_heavy_cols), file)
    
def load_static_models(epv_path='./data/static/EPV_grid.csv', trans_path='./data/static/Transition_gauss.csv'):
    """EPV 그리드와 Transition 모델을 로드합니다."""
    # EPV Load
    if not os.path.exists(epv_path):
        raise FileNotFoundError(f"EPV model not found at {epv_path}")
    EPV = np.loadtxt(epv_path, delimiter=',')
    EPV = EPV / np.max(EPV)

    # Transition Load
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Transition model not found at {trans_path}")
    Trans_df = pd.read_csv(trans_path, header=None)
    Trans = np.array(Trans_df)
    Trans = Trans / np.max(Trans)
    
    return EPV, Trans

def run_obso_events(loader, n_jobs=1):
    # read in event data
    events = loader.get_event_data()
    
    # read in tracking data
    tracking_home = loader.get_trace_data("Home")
    tracking_away = loader.get_trace_data("Away")

    # consistent direction of play
    events = loader.to_single_playing_direction(events)
    tracking_home = loader.to_single_playing_direction(tracking_home)
    tracking_away = loader.to_single_playing_direction(tracking_away)

    # calculate player velocities
    tracking_home = loader.calc_player_velocities(tracking_home)
    tracking_away = loader.calc_player_velocities(tracking_away)

    # event data convert common format 
    events_df = loader.convert_common_format_for_event(events, tracking_home, tracking_away)

    # check 'Home' team in tracking and event data
    events_df = loader.check_home_away_event(events_df, tracking_home, tracking_away)
    
    tracking_home = tracking_home.set_index('frame_id')
    tracking_away = tracking_away.set_index('frame_id')
    assert len(tracking_home) == len(tracking_away), f"Home and Away tracking data must have the same number of frames: {len(tracking_home)} vs {len(tracking_away)}"
    
    # set parameter and static model
    EPV, Transition = load_static_models()
    params = pc.default_model_params()
    GK_numbers = [loader.find_goalkeeper(tracking_home), loader.find_goalkeeper(tracking_away)]

    # calculate pitch control for all events.   
    if n_jobs == 1: # single processing
        PPCF = []
        for (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in tqdm(zip(events_df.groupby('period_id'),tracking_home.groupby('period_id'),tracking_away.groupby('period_id')), desc="Calculating Pitch Control for Events"):
            for pass_row in tqdm(period_events.itertuples()):
                ppcf, _, _ = pc.generate_pitch_control_for_event(pass_row, 
                                                                 period_tracking_home.loc[pass_row.start_frame], period_tracking_away.loc[pass_row.start_frame], 
                                                                 params, GK_numbers, offsides=True, field_dimen=(105,68))
                PPCF.append((pass_row.period_id, pass_row.Index, ppcf))
    else: # multiprocessing
        def process_single_event_wrapper(pass_row, period_tracking_home, period_tracking_away, 
                                         params, GK_numbers, offsides=True, field_dimen=FIELD_DIMS):
            ppcf, _, _ = pc.generate_pitch_control_for_event(pass_row, period_tracking_home, period_tracking_away, 
                                                             params, GK_numbers, offsides=offsides, field_dimen=field_dimen)
            return pass_row.period_id, pass_row.Index, ppcf

        with tqdm_joblib(tqdm(desc="Calculating Pitch Control for Events", total=len(events_df))):
            PPCF = Parallel(n_jobs=n_jobs, backend="loky", batch_size=256)(
                    delayed(process_single_event_wrapper)(
                        pass_row,
                        period_tracking_home.loc[pass_row.start_frame],
                        period_tracking_away.loc[pass_row.start_frame],
                        params,
                        GK_numbers,
                        offsides=True,
                        field_dimen=(105,68),
                    )
                for (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in zip(events_df.groupby('period_id'),tracking_home.groupby('period_id'),tracking_away.groupby('period_id'))
                for pass_row in period_events.itertuples()
            )
        PPCF.sort(key=lambda x: (x[0], x[1])) # Sort by Event Index
    PPCF_dict = {(period_id, event_index): ppcf for period_id, event_index, ppcf in PPCF} 

    # calculate OBSO
    obso_df, home_obso_df, away_obso_df, home_onball_obso_df, away_onball_obso_df = [], [], [], [], []
    for (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in zip(events_df.groupby('period_id'),tracking_home.groupby('period_id'),tracking_away.groupby('period_id')):
        obso = np.zeros((len(period_events), 32, 50))
        for event_num, row in enumerate(period_events.itertuples()):
            if row.team == 'Home':
                direction = loader.find_playing_direction(period_tracking_home, 'Home')
            elif row.team == 'Away': 
                direction = loader.find_playing_direction(period_tracking_away, 'Away')
            else:
                raise ValueError(f"Unknown team for event {event_num}: {row}")
            
            obso[event_num], ppcf, trans, score = obs.calc_obso(PPCF_dict[(row.period_id, row.Index)], Transition, EPV, 
                                                                ball_start_pos=(row.start_x, row.start_y), 
                                                                attack_direction=direction)
            obso_df.append(
            {
                    "event_number": event_num,
                    "event_frame": row.start_frame,
                    "period_id": row.period_id,
                    "team": row.team,
                    "ball_x": row.start_x,
                    "ball_y": row.start_y,
                    "obso_map": obso[event_num],
                    "ppcf_map": ppcf,
                    "trans_map": trans,
                    "score_map": score
                }
            )
     
        # calculate player evaluate OBSO and on-ball OBSO
        home_obso, away_obso = obs.calc_player_evaluate_event(obso, period_events, period_tracking_home, period_tracking_away)
        home_onball_obso, away_onball_obso = obs.calc_onball_obso(period_events, period_tracking_home, period_tracking_away, home_obso, away_obso)      
        
        for data in [home_obso, away_obso, home_onball_obso, away_onball_obso]:
            data['period_id'] = row.period_id

        home_obso_df.append(home_obso)
        away_obso_df.append(away_obso)
        home_onball_obso_df.append(home_onball_obso)
        away_onball_obso_df.append(away_onball_obso)

    obso_df = pd.DataFrame(obso_df)
    home_obso_df = pd.concat(home_obso_df, ignore_index=True)
    away_obso_df = pd.concat(away_obso_df, ignore_index=True)
    home_onball_obso_df = pd.concat(home_onball_obso_df, ignore_index=True)
    away_onball_obso_df = pd.concat(away_onball_obso_df, ignore_index=True)

    return obso_df, home_obso_df, away_obso_df, home_onball_obso_df, away_onball_obso_df

def run_obso_traces(loader, n_jobs=1):
    # read in tracking data
    tracking_home = loader.get_trace_data("Home")
    tracking_away = loader.get_trace_data("Away")

    # consistent direction of play
    tracking_home = loader.to_single_playing_direction(tracking_home)
    tracking_away = loader.to_single_playing_direction(tracking_away)

    # calculate player velocities
    tracking_home = loader.calc_player_velocities(tracking_home)
    tracking_away = loader.calc_player_velocities(tracking_away)

    # filter only alive ball state and valid team
    tracking_home = tracking_home[(tracking_home["ball_state"] == "alive") & (tracking_home["team"].isin(["Home", "Away"]))].reset_index(drop=True)
    tracking_away = tracking_away[(tracking_away["ball_state"] == "alive") & (tracking_away["team"].isin(["Home", "Away"]))].reset_index(drop=True)
    tracking_home = tracking_home.set_index('frame_id')
    tracking_away = tracking_away.set_index('frame_id')
    assert len(tracking_home) == len(tracking_away), f"Home and Away tracking data must have the same number of frames: {len(tracking_home)} vs {len(tracking_away)}"

    # set parameter and static model
    EPV, Transition = load_static_models()
    params = pc.default_model_params()
    GK_numbers = [loader.find_goalkeeper(tracking_home), loader.find_goalkeeper(tracking_away)]

    # calculate OBSO for all traces
    if n_jobs == 1: # single processing
        PPCF = []
        for (_, period_tracking_home), (_, period_tracking_away) in tqdm(zip(tracking_home.groupby('period_id'),tracking_away.groupby('period_id')), desc="Calculating Pitch Control for Traces"):
            for row in period_tracking_home.itertuples():
                ppcf, _, _ = pc.generate_pitch_control_for_trace(period_tracking_home.loc[row.Index], period_tracking_away.loc[row.Index], 
                                                                 params, GK_numbers, offsides=True, field_dimen=(105,68))
                PPCF.append((row.period_id, row.Index, ppcf))
    else: # multiprocessing
        def process_single_trace_wrapper(period_tracking_home, period_tracking_away, 
                                         params, GK_numbers, offsides=True, field_dimen=FIELD_DIMS):
            ppcf, _, _ = pc.generate_pitch_control_for_trace(period_tracking_home, period_tracking_away, 
                                                             params, GK_numbers, offsides=offsides, field_dimen=field_dimen)
            return period_tracking_home.period_id, period_tracking_home.name, ppcf

        with tqdm_joblib(tqdm(desc="Calculating Pitch Control for Traces", total=len(tracking_home))):
            PPCF = Parallel(n_jobs=n_jobs, backend="loky", batch_size=256)(
                    delayed(process_single_trace_wrapper)(
                        period_tracking_home.loc[row.Index],
                        period_tracking_away.loc[row.Index],
                        params,
                        GK_numbers,
                        offsides=True,
                        field_dimen=(105,68),
                    )
                for (_, period_tracking_home), (_, period_tracking_away) in zip(tracking_home.groupby('period_id'), tracking_away.groupby('period_id'))
                for row in period_tracking_home.itertuples()
            )
        PPCF.sort(key=lambda x: (x[0], x[1])) # Sort by Event Index
    PPCF_dict = {(period_id, trace_index): ppcf for period_id, trace_index, ppcf in PPCF} 

    # calculate OBSO
    obso_df, home_obso_df, away_obso_df = [], [], []
    for (_, period_tracking_home), (_, period_tracking_away) in zip(tracking_home.groupby('period_id'),tracking_away.groupby('period_id')):
        obso = np.zeros((len(period_tracking_home), 32, 50))
        for trace_num, row in enumerate(period_tracking_home.itertuples()):
            if row.team =='Home':
                direction = loader.find_playing_direction(period_tracking_home, 'Home')
            elif row.team =='Away': 
                direction = loader.find_playing_direction(period_tracking_away, 'Away')
            else:
                print(f"Unknown team for trace {trace_num}")
                continue
            
            obso[trace_num], ppcf, trans, score = obs.calc_obso(PPCF_dict[(row.period_id, row.Index)], Transition, EPV, 
                                                                ball_start_pos=(tracking_home.loc[row.Index, "ball_x"], tracking_home.loc[row.Index, "ball_y"]), 
                                                                attack_direction=direction)
            obso_df.append(
                {
                    "trace_number": trace_num,
                    "trace_frame": row.Index,
                    "period_id": row.period_id,
                    "team": row.team,
                    "ball_x": row.ball_x,
                    "ball_y": row.ball_y,
                    "obso_map": obso[trace_num],
                    "ppcf_map": ppcf,
                    "trans_map": trans,
                    "score_map": score
                }
            )

        # calculate player evaluate OBSO
        home_obso, away_obso = obs.calc_player_evaluate_trace(obso, period_tracking_home, period_tracking_away)

        for data in [home_obso, away_obso]:
            data['period_id'] = row.period_id

        home_obso_df.append(home_obso)
        away_obso_df.append(away_obso)

    obso_df = pd.DataFrame(obso_df)
    home_obso_df = pd.concat(home_obso_df, ignore_index=True)
    away_obso_df = pd.concat(away_obso_df, ignore_index=True)

    return obso_df, home_obso_df, away_obso_df

def run_obso_virtual(loader, n_jobs=1):
    # read in event data
    events = loader.get_event_data()

    # read in tracking data
    tracking_home = loader.get_trace_data("Home")
    tracking_away = loader.get_trace_data("Away")

    # consistent direction of play
    events = loader.to_single_playing_direction(events)
    tracking_home = loader.to_single_playing_direction(tracking_home)
    tracking_away = loader.to_single_playing_direction(tracking_away)

    # calculate player velocities
    tracking_home = loader.calc_player_velocities(tracking_home)
    tracking_away = loader.calc_player_velocities(tracking_away)

    # event data convert common format 
    events_df = loader.convert_common_format_for_event(events, tracking_home, tracking_away)

    # check 'Home' team in tracking and event data
    events_df = loader.check_home_away_event(events_df, tracking_home, tracking_away)

    tracking_home = tracking_home.set_index('frame_id')
    tracking_away = tracking_away.set_index('frame_id')
    assert len(tracking_home) == len(tracking_away), f"Home and Away tracking data must have the same number of frames: {len(tracking_home)} vs {len(tracking_away)}"

    # set parameter and static model
    EPV, Transition = load_static_models()
    params = pc.default_model_params()
    GK_numbers = [loader.find_goalkeeper(tracking_home), loader.find_goalkeeper(tracking_away)]

    # filter pass events with successful receive event by same team
    pass_events_df = events_df[
        (events_df['type_name'].isin(PASS_EVENTS)) & 
        (events_df["result_name"] == True) & 
        (events_df["team"] == events_df["receiver_team"]) &
        (events_df["receive_frame"].notna())
    ].reset_index(drop=True)
    
    if n_jobs == 1: # single processing
        virtual_results = []
        for (_, period_pass_events), (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in tqdm(zip(pass_events_df.groupby('period_id'),events_df.groupby('period_id'),tracking_home.groupby('period_id'),tracking_away.groupby('period_id')), desc="Calculating Virtual OBSO"):
            for _, pass_row in period_pass_events.iterrows():
                # Find corresponding receive event
                receive_event = period_events[
                    (period_events['type_name'].isin(RECEIVE_EVENTS)) & 
                    (pass_row.start_frame - BEFORE_MARGIN_FRAME <= period_events['start_frame']) & 
                    (period_events['start_frame'] <= pass_row.start_frame) & 
                    (period_events['period_id'] == pass_row.period_id) & 
                    (period_events['player_code'] == pass_row.player_code)
                ]

                pass_frame = int(pass_row.start_frame)
                receive_frame = int(pass_row.start_frame - BEFORE_MARGIN_FRAME) if receive_event.empty else int(receive_event.iloc[-1].start_frame)

                # 2. Generate Ghost Trajectory
                period_tracking_home_ghost = obs.generate_virtual_trajectory(period_tracking_home, pass_frame, receive_frame, 
                                                                             before_margin_frame=BEFORE_MARGIN_FRAME, after_margin_frame=AFTER_MARGIN_FRAME)
                period_tracking_away_ghost = obs.generate_virtual_trajectory(period_tracking_away, pass_frame, receive_frame, 
                                                                             before_margin_frame=BEFORE_MARGIN_FRAME, after_margin_frame=AFTER_MARGIN_FRAME)

                # 3. Calculate Pitch Control for ALL frames in Ghost Trajectory
                ppcf_list = []
                for row in period_tracking_home_ghost.itertuples():
                    ppcf, _, _ = pc.generate_pitch_control_for_virtual(pass_row, period_tracking_home_ghost.loc[row.Index], period_tracking_away_ghost.loc[row.Index], 
                                                                     params, GK_numbers, offsides=True, field_dimen=(105,68))
                    ppcf_list.append(ppcf)

                virtual_results.append((pass_row, period_tracking_home_ghost, period_tracking_away_ghost, ppcf_list))
    else: # multiprocessing
        def process_single_virtual_wrapper(pass_row, period_events, 
                                         period_tracking_home, period_tracking_away, 
                                         params, GK_numbers, offsides=True, field_dimen=FIELD_DIMS):
            receive_event = period_events[
                (period_events['type_name'].isin(RECEIVE_EVENTS)) & 
                (pass_row.start_frame - BEFORE_MARGIN_FRAME <= period_events['start_frame']) & 
                (period_events['start_frame'] <= pass_row.start_frame) & 
                (period_events['period_id'] == pass_row.period_id) & 
                (period_events['player_code'] == pass_row.player_code)
            ]
            
            pass_frame = int(pass_row.start_frame)
            receive_frame = int(pass_row.start_frame - BEFORE_MARGIN_FRAME) if receive_event.empty else int(receive_event.iloc[-1].start_frame)

            # 2. Generate Ghost Trajectory
            period_tracking_home_ghost = obs.generate_virtual_trajectory(period_tracking_home, pass_frame, receive_frame, 
                                                                         before_margin_frame=BEFORE_MARGIN_FRAME, after_margin_frame=AFTER_MARGIN_FRAME)
            period_tracking_away_ghost = obs.generate_virtual_trajectory(period_tracking_away, pass_frame, receive_frame, 
                                                                         before_margin_frame=BEFORE_MARGIN_FRAME, after_margin_frame=AFTER_MARGIN_FRAME)
            # 3. Calculate Pitch Control for ALL frames in Ghost Trajectory
            ppcf_list = []
            for row in period_tracking_home_ghost.itertuples():
                ppcf, _, _ = pc.generate_pitch_control_for_virtual(pass_row, period_tracking_home_ghost.loc[row.Index], period_tracking_away_ghost.loc[row.Index], 
                                                                params, GK_numbers, offsides=offsides, field_dimen=field_dimen)
                ppcf_list.append(ppcf)

            return pass_row, period_tracking_home_ghost, period_tracking_away_ghost, ppcf_list

        with tqdm_joblib(tqdm(desc=f"Calculating Virtual OBSO: {loader.game_id}", total=len(pass_events_df))):
            virtual_results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=256)(
                    delayed(process_single_virtual_wrapper)(
                        pass_row,
                        period_events,
                        period_tracking_home,
                        period_tracking_away,
                        params,
                        GK_numbers,
                        offsides=True,
                        field_dimen=(105,68),
                    )
                for (_, period_pass_events), (_, period_events), (_, period_tracking_home), (_, period_tracking_away) in zip(pass_events_df.groupby('period_id'), events_df.groupby('period_id'), tracking_home.groupby('period_id'), tracking_away.groupby('period_id'))
                for _, pass_row in period_pass_events.iterrows()
            )
        virtual_results.sort(key=lambda x: (x[0].period_id, x[0].name)) # Sort by Event Index
    
    # calculate OBSO from virtual results
    pass_df, virtual_obso_df, virtual_home_tracking, virtual_away_tracking = [], [], [], []
    for result in virtual_results:
        pass_row, tracking_home_ghost, tracking_away_ghost, ppcf = result

        obso = np.zeros((len(tracking_home_ghost), 32, 50))
        trans = np.zeros((len(tracking_home_ghost), 32, 50))
        score = np.zeros((len(tracking_home_ghost), 32, 50))
        for trace_num, row in enumerate(tracking_home_ghost.itertuples()):
            # checking attacking direction
            if pass_row.team =='Home':
                direction = loader.find_playing_direction(tracking_home_ghost, 'Home')
                ball_start_pos=(tracking_home_ghost.loc[row.Index, f"{pass_row.player_code}_x"], tracking_home_ghost.loc[row.Index, f"{pass_row.player_code}_y"])
            elif pass_row.team =='Away':
                direction = loader.find_playing_direction(tracking_away_ghost, 'Away')
                ball_start_pos=(tracking_away_ghost.loc[row.Index, f"{pass_row.player_code}_x"], tracking_away_ghost.loc[row.Index, f"{pass_row.player_code}_y"])
            else:
                raise ValueError(f"Invalid attacking_team: {pass_row.team}")

            obso[trace_num], _, trans[trace_num], score[trace_num] = obs.calc_obso(ppcf[trace_num], Transition, EPV, 
                                                                                   ball_start_pos=ball_start_pos,
                                                                                   attack_direction=direction)
        
        obso_df = obs.calc_player_evaluate_virtual(pass_row.team, obso, tracking_home_ghost, tracking_away_ghost)
        obso_df['period_id'] = pass_row.period_id
        obso_df['obso_map'] = [obso[i] for i in range(obso.shape[0])]
        obso_df['ppcf_map'] = [ppcf[i] for i in range(len(ppcf))]
        obso_df['trans_map'] = [trans[i] for i in range(len(trans))]
        obso_df['score_map'] = [score[i] for i in range(len(score))]

        for data in [pass_row, obso_df, tracking_home_ghost, tracking_away_ghost]:
            data['event_id'] = pass_row.name
            
        pass_df.append(pass_row)
        virtual_obso_df.append(obso_df)
        virtual_home_tracking.append(tracking_home_ghost.reset_index(drop=False).rename(columns={"index": "frame_id"})) # frame_id reconstruct
        virtual_away_tracking.append(tracking_away_ghost.reset_index(drop=False).rename(columns={"index": "frame_id"})) # frame_id reconstruct

    pass_df = pd.DataFrame(pass_df)
    virtual_obso_df = pd.concat(virtual_obso_df, ignore_index=True) 
    virtual_home_tracking = pd.concat(virtual_home_tracking, ignore_index=True)
    virtual_away_tracking = pd.concat(virtual_away_tracking, ignore_index=True)

    return pass_df, virtual_obso_df, virtual_home_tracking, virtual_away_tracking

if __name__ == "__main__":
    """
        python calculate_obso.py --provider bepro --unit event --game_id 153373 --data_dir ./data/bepro/elastic --output_dir ./data/bepro/obso --n_jobs -1
        python calculate_obso.py --provider bepro --unit trace --game_id 126476 --data_dir ./data/bepro/elastic --output_dir ./data/bepro/obso --n_jobs -1
        python calculate_obso.py --provider bepro --unit virtual --game_id 153381 --data_dir ./data/bepro/elastic --output_dir ./data/bepro/obso --n_jobs -1

        python calculate_obso.py --provider dfl --unit event --game_id DFL-MAT-J03WMX --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1
        python calculate_obso.py --provider dfl --unit trace --game_id DFL-MAT-J03WMX --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1
        python calculate_obso.py --provider dfl --unit virtual --game_id all --data_dir ./data/dfl/elastic --output_dir ./data/dfl/obso --n_jobs -1
    """
    parser = argparse.ArgumentParser(description="Calculate OBSO based on Pitch Control")
    
    parser.add_argument('--provider', type=str, required=True, choices=['bepro', 'dfl'],
                        help='Data provider name (e.g., bepro, dfl)')
    parser.add_argument('--unit', type=str, default='event', choices=['trace', 'event', 'virtual'],
                        help='Calculation unit: "trace" (all frames) or "event" (event frames only) or "virtual" (pass event frames with virtual trajectory)')
    parser.add_argument('--game_id', type=str, default='all',
                        help='Specific game ID to process, or "all" to process all games in directory')
    parser.add_argument('--data_dir', type=str, default='./data/bepro/processed',
                        help='Root directory for input data')
    parser.add_argument('--output_dir', type=str, default='./data/bepro/obso',
                        help='Directory to save output files')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for processing (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Get Game IDs
    game_ids = os.listdir(path=args.data_dir) if args.game_id == 'all' else [args.game_id]
    # game_ids = [id for id in game_ids if not os.path.exists(f"{args.output_dir}/{id}/{args.unit}")]
    game_ids = [id for id in game_ids]
    print(f"Game IDs: {game_ids}")
    print(f"Available CPU: {os.cpu_count()}")

    # 2. Process Games
    exception_games = []
    print(f"---- Starting OBSO calculation for unit: {args.unit} ----\n")
    if args.unit == 'event':
        for game_id in tqdm(game_ids, desc="Processing Games for Event-based OBSO"):
            print(f"Processing game {game_id} for event OBSO...")
            loader = get_loader(args.provider, args.data_dir, game_id)
            obso, home_obso, away_obso, home_onball_obso, away_onball_obso = run_obso_events(loader, n_jobs=args.n_jobs)

            save_obso(
                results={
                    "obso": obso,
                    "home_obso": home_obso,
                    "away_obso": away_obso,
                    "home_onball_obso": home_onball_obso,
                    "away_onball_obso": away_onball_obso
                }, 
                output_dir = f"{args.output_dir}/{game_id}/event"
            )
            print(f"Saved event OBSO for game {game_id}\n")
    elif args.unit == 'trace':
        for game_id in tqdm(game_ids, desc="Processing Games for Tracking-based OBSO"):
            print(f"Processing game {game_id} for trace OBSO...")
            loader = get_loader(args.provider, args.data_dir, game_id)
            obso, home_obso, away_obso = run_obso_traces(loader, n_jobs=args.n_jobs)

            save_obso(
                results={
                    "obso": obso,
                    "home_obso": home_obso,
                    "away_obso": away_obso
                }, 
                output_dir = f"{args.output_dir}/{game_id}/trace"
            )
            print(f"Saved trace OBSO for game {game_id}\n")
    elif args.unit == 'virtual':
        for game_id in tqdm(game_ids, desc=f"Processing Games for Virtual-based OBSO"):
            if os.path.exists(f"{args.output_dir}/{game_id}/virtual"):
                print(f"Virtual OBSO already exists for game {game_id}, skipping...")
                continue
            print(f"Processing game {game_id} for virtual OBSO...")
            loader = get_loader(args.provider, args.data_dir, game_id)

            #pass_df, virtual_obso_df, virtual_home_tracking, virtual_away_tracking = run_obso_virtual(loader, n_jobs=args.n_jobs)
            try:
                pass_df, virtual_obso_df, virtual_home_tracking, virtual_away_tracking = run_obso_virtual(loader, n_jobs=args.n_jobs)
            except Exception as e:
                print(f"Exception occurred while processing game {game_id}: {e}")
                exception_games.append(game_id)
                continue

            save_obso_compressed(
                results={
                    "pass_events": pass_df,
                    "virtual_obso": virtual_obso_df,
                    "virtual_home_tracking": virtual_home_tracking,
                    "virtual_away_tracking": virtual_away_tracking
                },
                output_dir = f"{args.output_dir}/{game_id}/virtual"
            )
            print(f"Saved virtual OBSO for game {game_id}\n")
    else:
        raise ValueError(f"Unknown calculation unit: {args.unit}")
    
    print(f"Exception games during virtual OBSO processing: {exception_games}")
    print("All processing complete.")