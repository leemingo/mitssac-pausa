"""bepro data to SPADL converter."""
from typing import Optional
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

bodyparts: list[str] = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]
results: list[str] = [
    "fail",
    "success",
    "offside",
    "owngoal",
    "yellow_card",
    "red_card",
]
actiontypes: list[str] = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "take_on",
    "foul",
    "tackle",
    "interception",
    "shot",
    "shot_penalty",
    "shot_freekick",
    "keeper_save",
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
    "clearance",
    "bad_touch",
    "non_action",
    "dribble",
    "goalkick",
]

COLS = [
    "period_id", "event_time", 
    "player_id", "player_name", "event_types",
    "x", "y", "to_x", "to_y",
]

def print_events(cond_events: pd.DataFrame) -> None:
    print("\n====================")
    print(cond_events[COLS])
    # for i in cond_events.index:
    #     print(f"Index: {i}, Event Types: {cond_events.at[i, 'event_types']}")

def convert_to_actions(
    events: pd.DataFrame, 
	xy_fidelity_version: Optional[int] = None,
	shot_fidelity_version: Optional[int] = None,
) -> DataFrame:
    """
    Convert K-league events to SPADL actions.
    """

    events["period_id"] = events["period_order"] + 1 
    events = events.rename(columns={"events": "event_types"})
    events = events.sort_values(["period_id", "event_time"], kind="mergesort").reset_index(drop=True) 
    
    # 분석에 활용하지 않은 이벤트 제거: 결측치 & 중복값
    events = _clean_events(events, remove_event_types=["HIR", "MAX_SPEED", "VHIR", 
                                                       "Assists", "Key Passes", "SPRINT", 
                                                       "Set Piece Defence", "Passes Received", "Turnover", 
                                                       "Duels", "Blocks", "Crosses Received",
                                                       "Recoveries"])
    events = _fix_offside(events)
    events = _fix_defensive_line_support(events)    # Convert defensive_line_support to tackle or interception

    # 공격 이벤트와 수비 이벤트가 동시에 발생한 경우 이를 분리합니다.
    events = insert_defensive_actions(events, defensive_action="Interceptions")
    events = insert_defensive_actions(events, defensive_action="Tackles")

    # K-league데이터셋 형태의 aciton을 SPALD형태로 변환
    events["type_name"] = events["event_types"].apply(_get_type_name)

    # 각 액션에 대해 액션 ID, 터치 부위, 결과, 끝 위치를 정의함
    events[["type_id", "bodypart_id", "result_id"]] = events.apply(_parse_event, axis=1, result_type="expand")

    actions = pd.DataFrame()
    actions["game_id"] = events.match_id.astype(int)
    actions["original_event_id"] = events.event_id
    actions["period_id"] = events.period_id.astype(int)

    # convert milliseconds to seconds
    # First half kick-off: 0(ms), second half kick-off: 2,700,000(ms)
    #actions["utc_timestamp"] = events["utc_timestamp"]
    actions["time_seconds"] = (
        events["event_time"] * 0.001 
        - ((events.period_id > 1) * 45 * 60) # convert 45(minutes) to 45*60(seconds)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )

    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id
    actions["object_id"] = events.object_id

    # K-league 경기장 형태를 SPADL형태로 변환 : 68x105 -> 105x68로 변환
    actions["start_x"] = events.x
    actions["start_y"] = events.y
    actions["end_x"] = events.to_x
    actions["end_y"] = events.to_y

    actions["type_name"] = events.type_name
    actions["type_id"] = events.type_id.astype(int)

    actions["bodypart_name"] = events.bodypart_id.apply(lambda id: bodyparts[id])
    actions["bodypart_id"] = events.bodypart_id.astype(int)

    actions["result_name"] = events.result_id.apply(lambda id: results[id])
    actions["result_id"] = events.result_id.astype(int)
    actions["success"] = events.result_id.apply(lambda id: id == results.index("success"))

    actions = (
        actions[actions.type_id != actiontypes.index("non_action")]
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    actions = _fix_dribble(actions)  # dribble의 끝 위치를 조정
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions) # TODO: _add_dribbles_after_receive가 더 정확하게 드리블 생성 가능함. 단, baseline을 위해 제외함

    return actions

def _clean_events(df_events: pd.DataFrame, remove_event_types) -> pd.DataFrame:
    """
    데이터프레임에서 특정 이벤트 타입을 제거하는 함수.
    remove_event_types (list): 제거할 이벤트 타입 목록

    - 결측치 조건(missing_cond)
    1.24	[Duel]	[Aerial]	...	NaN	NaN	NaN	NaN	NaN	NaN	[{'event_type': 'Duel', 'sub_event_type': 'Aerial...	NaN	NaN	35
    67	85848	94916404	1	4641.0	259769.0	163181	0.193699	0.5342 event_types이 기록되어 있지 않는 경우 -> parsing이 불가능함, 단순히 이전 정보만으로는 예측 불가능
    2. event_types이 기록되어 있는데, team_id & player_id정보가 기록되어 있지 않는 경우 -> 이전 정보로는 불가능하겠지만, 다음 정보로는 가능함.
    
    - 중복 데이터(duplicated_cond)
    1. event_id가 중복되는 경우 -> 제거
    2. event_id는 다른데 그 외 데이터가 중복되는 경우 -> 제거
    """

    # 해당 이벤트만 제거
    # ex) Pass + Control Under Pressure -> Pass 
    df_events["event_types"] = df_events["event_types"].apply(
        lambda event_list: [event for event in event_list if ("event_name" in event.keys()) and (event["event_name"] not in remove_event_types)]
    )

    # 처음부터 빈 리스트이거나 제거하므로써 remove_event_types로 인해 빈 리스트가 된 행 제거
    missing_cond = (
        (df_events['event_types'].apply(len) == 0)
        # | (events["team_id"].isna())   # we do not remove missing team_id to better capture the context
        # | (events["player_id"].isna()) # we do not remove missing player_id to better capture the context
    )
    df_events = df_events[~missing_cond].reset_index(drop=True)

    # keep=fist: 중복된 데이터 중 첫번째 데이터만 남기고 나머지 제거(첫번째 데이터만 False)
    df = df_events.copy()

    # duplicated_cond2: event_id는 다른데 그 외 데이터가 모두 중복되는 경우
    event_cols = df_events.columns.tolist()
    for col in event_cols:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) or isinstance(x, dict) else x) # duplicated함수는 list, dict를 지원하지 않음
    duplicated_cond = df.duplicated(subset=event_cols, keep="first") # 첫번째 데이터만 False -> ~False=True
    
    return df_events[~duplicated_cond].reset_index(drop=True)

def shift_with_edge_fix(actions: pd.DataFrame, shift_value: int) -> pd.DataFrame:
    """
    Shift each group by a specified value and fill NaN only in the first or last row.
    Does not alter NaN values in the middle of the group to avoid affecting naturally missing data.
    """

    shift_action = actions.groupby("period_id").shift(shift_value)

    if shift_value < 0:
        # When shifting upwards, last row gets NaN, so fill it with the original last value
        fill_indices = actions.groupby("period_id").tail(abs(shift_value)).index
    else:
        # When shifting downwards, first row gets NaN, so fill it with the original first value
        fill_indices = actions.groupby("period_id").head(abs(shift_value)).index

    # Fill the NaN rows with the corresponding original values
    shift_action.loc[fill_indices] = actions.loc[fill_indices]
    shift_action["period_id"] = actions["period_id"]

    return shift_action

def _parse_event(event : pd.Series) -> tuple[int, int, float, float]:
    # 23 possible values : pass, cross, throw-in, 
    # crossed free kick, short free kick, crossed corner, short corner, 
    # take-on, foul, tackle, interception, 
    # shot, penalty shot, free kick shot, 
    # keeper save, keeper claim, keeper punch, keeper pick-up, 
    # clearance, bad touch, dribble and goal kick.
    events = {
        "pass": _parse_pass_event,
        "cross": _parse_pass_event,
        "throw_in": _parse_pass_event,
        "freekick_crossed": _parse_pass_event,
        "freekick_short": _parse_pass_event,
        "corner_crossed": _parse_pass_event,
        "corner_short": _parse_pass_event,

        "take_on": _parse_take_on_event,

        "foul": _parse_foul_event,

        "tackle" : _parse_tackle_event,

        "interception": _parse_interception_event,

        "shot": _parse_shot_event,
        "shot_penalty": _parse_shot_event,
        "shot_freekick": _parse_shot_event,

        "keeper_save" : _parse_goalkeeper_event,
        "keeper_claim" : _parse_goalkeeper_event,
        "keeper_punch" : _parse_goalkeeper_event,
        "keeper_pick_up" : _parse_goalkeeper_event,
        "Defensive_Line_Support" : _parse_goalkeeper_event,

        "clearance" : _parse_clearance_event,
        "bad_touch" : _parse_bad_touch_event,
        "dribble" : _parse_dribble_event,

        "goalkick" : _parse_pass_event,
    }

    parser = events.get(event["type_name"], _parse_event_as_non_action)
    bodypart, result = parser(event["event_types"])

    actiontype = actiontypes.index(event["type_name"])
    bodypart = bodyparts.index(bodypart)
    result = results.index(result)
    
    return actiontype, bodypart, result

def _get_type_name(event_types: list) -> str:
    if any(e["event_name"] == "Crosses" for e in event_types):
        set_piece_dict = next(
            (e for e in event_types
            if e.get("event_name") == "Set Pieces"),
            None
        )
        if set_piece_dict is None:
            a = "cross"
        else:
            if set_piece_dict.get("property", {}).get("Type") == "Freekicks":
                a = "freekick_crossed"
            elif set_piece_dict.get("property", {}).get("Type") == "Corners":
                a = "corner_crossed"
            else:
                raise ValueError(f"Unknown set piece type in event_types: {event_types}")
    elif any(e["event_name"] == "Passes" for e in event_types):
        set_piece_dict = next(
            (e for e in event_types
            if e.get("event_name") == "Set Pieces"),
            None
        )
        if set_piece_dict is None:
            a = "pass"
        else:
            if set_piece_dict.get("property", {}).get("Type") == "Freekicks":
                a = "freekick_short"
            elif set_piece_dict.get("property", {}).get("Type") == "Corners":
                a = "corner_short"
            elif set_piece_dict.get("property", {}).get("Type") == "Throw-Ins":
                a = "throw_in"
            elif set_piece_dict.get("property", {}).get("Type") == "Goal Kicks":
                a = "goalkick"
            else:
                raise ValueError(f"Unknown set piece type in event_types: {event_types}")
    elif any(e["event_name"] == "Shots & Goals" for e in event_types):
        set_piece_dict = next(
            (e for e in event_types
            if e.get("event_name") == "Set Pieces"),
            None
        )
        if set_piece_dict is None:
            a = "shot"
        else:
            if set_piece_dict.get("property", {}).get("Type") == "Freekicks":
                a = "shot_freekick"
            elif set_piece_dict.get("property", {}).get("Type") == "Penalty Kicks":
                a = "shot_penalty"
            else:
                raise ValueError(f"Unknown set piece type in event_types: {event_types}")
    elif any(e["event_name"] == "Take-on" for e in event_types):
        a = "take_on"
    elif any(e["event_name"] == "Step-in" for e in event_types): # API 2025: rename Carry to Step-in
        a = "dribble"
    elif any(e["event_name"] == "Saves" for e in event_types): # 골키퍼 액션 : Save, Aerial Clearnce, Defensive Line Support Succeeded
        save_dict = next(e for e in event_types if e.get("event_name") == "Saves")
        if save_dict.get("property", {}).get("Type") == "Catches":
            a = "keeper_save"
        elif save_dict.get("property", {}).get("Type") == "Parries":
            a = "keeper_punch"
        else:
            raise ValueError(f"Unknown save type in event_types: {event_types}")
    elif any(e["event_name"] == "Aerial Control" for e in event_types):  
        aerial_control_dict = next(e for e in event_types if e.get("event_name") == "Aerial Control")
        # 실패한 Aerial Clearance는 공의 방향에 영향을 미치지 않으므로 non_action으로 처리
        if aerial_control_dict.get("property", {}).get("Outcome") == "Succeeded":
            a = "keeper_claim"
        else:
            a = "non_action"
    elif any(e["event_name"] == "Clearances" for e in event_types):
        a = "clearance"
    elif any(e["event_name"] == "Fouls" for e in event_types):
        foul_dict = next(e for e in event_types if e.get("event_name") == "Fouls")
        if foul_dict.get("property", {}).get("Type") in ["Fouls", "Yellow Cards", "Red Cards", "Handball Foul", "Foul Throw"]:
            a = "foul"
        elif foul_dict.get("property", {}).get("Type") == "Fouls Won":
            a = "non_action" # 상대팀의 파울 유도는 non_action으로 정의
        elif foul_dict.get("property", {}).get("Penalty Kick Won") == "True":
            a = "non_action" 
        else:
            raise ValueError(f"Unknown foul type in event_types: {event_types}")
    elif any(e["event_name"] == "Tackles" for e in event_types):
        a = "tackle"
    elif any(e["event_name"] == "Interceptions" for e in event_types):
        a = "interception"
    elif any(e["event_name"] == "Mistakes" for e in event_types):
        a = "bad_touch"
    elif any(e["event_name"] == "Own Goals" for e in event_types):
        a = "bad_touch"
    else:
        a = "non_action"
    
    return a

def _fix_defensive_line_support(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert Defensive_Line_Support events to interception"""  
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)
    cond_defensive_line_support = df_events["event_types"].apply(lambda x: any(e.get("event_name") == "Defensive Line Supports" for e in x))
    cond_tackle = df_events["event_types"].apply(lambda x: any(e.get("event_name") == "Tackles" for e in x))
    same_player = df_events.player_id == df_events_next.player_id

    cond_interception = cond_defensive_line_support & same_player & ~cond_tackle # 수비 액션 이후 공을 소유하는 경우, Interception으로 정의
    cond_tackle = cond_defensive_line_support & (~same_player | cond_tackle)  # 수비 액션 이후 공을 소유하지 않는 경우, Tackle로 정의

    df_events.loc[cond_interception , "event_types"] = df_events.loc[cond_interception , "event_types"].apply(
        lambda event_list: event_list + [{"event_name": "Interceptions"}]
    )
    df_events.loc[cond_tackle, "event_types"] = df_events.loc[cond_tackle, "event_types"].apply(
        lambda event_list: [
            {**e, 'event_name': 'Tackles', 'property': {'Outcome': e.get("property", {}).get("Outcome")}} if e.get("event_name") == "Defensive Line Supports" 
            else e 
            for e in event_list
        ]
    )

    return df_events

def _fix_offside(df_events: pd.DataFrame) -> pd.DataFrame:
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)

    cond_pass = df_events["event_types"].apply(
        lambda x: any(e.get("event_name") in ["Passes", "Crosses"] for e in x)
    )
    cond_set_piece = df_events["event_types"].apply(
        lambda x: any(e.get("property") in ["Corners", "Freekicks"] for e in x)
    )
    cond_next_offside = df_events_next["event_types"].apply(
        lambda x: any(e.get("event_name") == "Offsides" for e in x)
    )

    df_events.loc[cond_pass & cond_next_offside, "event_types"] = df_events.loc[cond_pass & cond_next_offside, "event_types"].apply(
        lambda event_list: [
            {**e, "property": {"Outcome": "offside"}} if e.get("event_name") in ["Passes", "Crosses"] 
            else e 
            for e in event_list
        ]
    )
    df_events.loc[cond_set_piece & cond_next_offside, "event_types"] = df_events.loc[cond_set_piece & cond_next_offside, "event_types"].apply(
        lambda event_list: [
            {**e, "property": {"Outcome": "offside"}} if e.get("property") in ["Corners", "Freekicks"] 
            else e 
            for e in event_list
        ]
    )
    
    return df_events

def _fix_dribble(df_actions: pd.DataFrame) -> pd.DataFrame:
    """
        _fix_dribble : Update the end position of dribble events based on their success or failure.
        If the dribble failed, the end position is set to the position of the next event.
        If the dribble succeeded, the end position is set to the position of the next event that is not a tackle.
    """

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1)

    failed_tackle = (
        (df_actions_next['type_id'] == actiontypes.index('tackle')) &
        (df_actions_next['result_id'] == results.index('fail'))
    )
    failed_defensive = (
        failed_tackle & 
        (df_actions.team_id != df_actions_next.team_id)
    )

    # next_actions: 실패한 태클이 아닌 다음 이벤트의 위치를 드리블의 끝 위치로 보간
    # ex) dribble(A팀) -> tackle(B팀, fail) -> pass(A팀)의 경우, dribble의 끝 위치는 pass의 시작 위치로 정의
    next_actions = df_actions_next.mask(failed_defensive)[["start_x", "start_y"]].bfill()

    cond_dribble = df_actions.type_id == actiontypes.index("dribble")

    df_actions.loc[cond_dribble, "end_x"] = next_actions.loc[cond_dribble, "start_x"].values
    df_actions.loc[cond_dribble, "end_y"] = next_actions.loc[cond_dribble, "start_y"].values

    return df_actions

def _fix_clearances(df_actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(df_actions, shift_value=-1)

    cond_clearance = df_actions.type_id == actiontypes.index("clearance")

    df_actions.loc[cond_clearance, "end_x"] = next_actions.loc[cond_clearance, "start_x"].values
    df_actions.loc[cond_clearance, "end_y"] = next_actions.loc[cond_clearance, "start_y"].values

    return df_actions   

def _add_dribbles(actions: pd.DataFrame,
                  min_dribble_length: float = 3.0,
                  max_dribble_length: float = 60.0,
                  max_dribble_duration: float = 10.0) -> pd.DataFrame:

    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")
    not_offensive_foul = same_team & (
        next_actions.type_id != actiontypes.index("foul")
    )
    not_headed_shot = (next_actions.type_id != actiontypes.index("shot")) & (
        next_actions.bodypart_id != bodyparts.index("head")
    )

    # bad_touch를 한 경우
    not_bad_touch = (next_actions.type_id != actiontypes.index("bad_touch")) 
    # 동일한 사람이 연속으로 드리블하는 상황X
    not_dribble = (next_actions.type_id != actiontypes.index("dribble")) & (
        next_actions.type_id != actiontypes.index("take_on")
    )

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_id == next_actions.period_id
    
    dribble_idx = (
        same_team
        & far_enough
        & not_too_far
        & same_phase
        & same_period
        & not_offensive_foul
        & not_headed_shot
        & not_bad_touch
        & not_dribble
    )

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    #dribbles["utc_timestamp"] = (prev.utc_timestamp + (nex.utc_timestamp - prev.utc_timestamp) / 2)
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    dribbles["team_id"] = nex.team_id
    dribbles["player_id"] = nex.player_id

    dribbles["start_x"] = prev.end_x
    dribbles["start_y"] = prev.end_y
    dribbles["end_x"] = nex.start_x
    dribbles["end_y"] = nex.start_y

    dribbles["bodypart_id"] = bodyparts.index("foot")
    dribbles["type_id"] = actiontypes.index("dribble")
    dribbles["result_id"] = results.index("success")

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
    actions["action_id"] = range(len(actions))
    return actions

# 공격 이벤트와 수비 이벤트가 함께 존재하는지 확인하는 함수
def insert_defensive_actions(df_events: pd.DataFrame, defensive_action : str) -> pd.DataFrame:
    """Insert defensive actions before offensive actions when both occur at the same time."""

    def is_attack_and_defense(event_types : list) -> bool:
        has_attack = any(e.get("event_name") in ["Passes", "Crosses", "Shots & Goals", "Take-on", "Step-in", "Clearances"] for e in event_types) # 공격 이벤트
        has_defense = any(e.get("event_name") == defensive_action for e in event_types)

        return has_attack and has_defense

    cond_attack_and_defense = df_events["event_types"].apply(is_attack_and_defense)
    df_events_defense = df_events[cond_attack_and_defense].copy()

    if not df_events_defense.empty:
        #df_events_defense["utc_timestamp"] = df_events_defense["utc_timestamp"] - pd.to_timedelta(1, unit='ms')
        df_events_defense["event_time"] -= 1e-3
        df_events_defense["event_types"] = df_events_defense["event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_name") == defensive_action]
        )
        df_events.loc[cond_attack_and_defense, "event_types"] = df_events.loc[cond_attack_and_defense, "event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_name") != defensive_action]
        )

        df_events = pd.concat([df_events_defense, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "event_time"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

    return df_events

def _parse_event_as_non_action(event_types: list) -> tuple[str, str]:
    bodypart = "other"
    result = "fail"
    return bodypart, result

def _parse_pass_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"
    pass_outcome =  next(
        (e.get("property", {}).get("Outcome") for e in event_types 
        if e.get('event_name') in ["Passes", "Crosses"]), 
        None
    )
    if pass_outcome == "Succeeded":
        result = "success"
    elif pass_outcome == "Failed":
        result = "fail"  # Offside situations are handled in _fix_offside
    elif pass_outcome == "offside":
        result = "offside"
    else:
        raise ValueError(f"Unexpected outcome value: {pass_outcome}")

    return bodypart, result

def _parse_take_on_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"

    take_on_outcome =  next(
        (e.get("property", {}).get("Outcome") for e in event_types if e.get('event_name') == 'Take-on'), 
        None
    )
    if take_on_outcome  == "Succeeded":
        result = "success"
    elif take_on_outcome  == "Failed":
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {take_on_outcome}")

    return bodypart, result

def _parse_foul_event(event_types: list) -> tuple[str, str]:
    foul_outcome = next(
        (e.get("property", {}).get("Type") for e in event_types if e.get('event_name') == 'Fouls'), 
        None    
    )
    if foul_outcome in ["Handball Foul", "Foul Throw"]:
        bodypart = "other" 
    else:
        bodypart = "foot"

    # foul은 여러 결과가 동시에 존재할 수 있음(경고+퇴장 등)
    if foul_outcome == "Red Cards":
        result = "red_card"
    elif foul_outcome == "Yellow Cards":
        result = "yellow_card"
    else:
        result = "fail"
    
    return bodypart, result

def _parse_tackle_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"

    tackle_outcome = next(
        (e.get("property", {}).get("Outcome") for e in event_types if e.get('event_name') == 'Tackles'), 
        None
    )

    if tackle_outcome in ["Tackle Succeeded: Possession", "Succeeded"]:
        result = "success"
    elif tackle_outcome in ["Tackle Succeeded: No Possession", "Tackle Failed", "Failed"]:
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {tackle_outcome}")

    return bodypart, result

def _parse_interception_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"
    result = "success"  

    return bodypart, result

def _parse_shot_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"

    shot_outcome = next(
        (e.get("property", {}).get("Outcome") for e in event_types if e.get("event_name") == "Shots & Goals"), 
        None
    )

    if shot_outcome == "Goals":
        result = "success"
    elif shot_outcome in ["Shots On Target", "Shots Off Target", "Blocked Shots", "Keeper Rush-outs"]:
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {event_types}")

    return bodypart, result

def _parse_goalkeeper_event(event_types: list) -> tuple[str, str]:
    bodypart = "other"

    # Determine the result based on the event type
    if any(e.get("event_name") == "Saves" for e in event_types): # Catch and parry actions are always successful
        result = "success"
    elif any(e.get("event_name") == "Aerial Control" for e in event_types): # Claim actions can be successful or unsuccessful
        aerial_clearance_outcome = next(
            (e.get("property", {}).get("Outcome") for e in event_types if e.get("event_name") == "Aerial Control"), 
            None
        )
        if aerial_clearance_outcome == "Succeeded": 
            result = "success"
        elif aerial_clearance_outcome  == "Failed":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {aerial_clearance_outcome}")
    else:
        raise ValueError(f'Unexpected event_types: {event_types}')

    return bodypart, result

def _parse_clearance_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"
    result = "success"

    return bodypart, result

def _parse_bad_touch_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"
    result = "owngoal" if any(e.get("event_name") == "Own Goals" for e in event_types) else "fail"

    return bodypart, result

def _parse_dribble_event(event_types: list) -> tuple[str, str]:
    bodypart = "foot"
    result = "success"

    return bodypart, result