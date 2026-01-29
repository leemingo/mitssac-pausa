import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from sync.elastic import ELASTIC
from tools.sportec_data import SportecData
from tools.bepro_data import BeproData

# multiprocessing set start method
def multi_processing_elastic(match_id_list, root_dir, elastic_dir, n_jobs=-1):
    def process_single_elastic_wrapper(args):
        match_id, root_dir, elastic_dir = args

        os.makedirs(os.path.join(elastic_dir, match_id), exist_ok=True)
        if os.path.exists(os.path.join(elastic_dir, match_id, "tracking.parquet")):
            print(f"Match {match_id} already processed. Skipping.")
            return

        try:
            if "bepro" in root_dir.lower():
                match = BeproData(root_dir, match_id, load_tracking=True)
            elif "dfl" in root_dir.lower():
                match = SportecData(root_dir, match_id, load_tracking=True)
            else:
                raise ValueError("Unknown data source. Please check the root_dir.")
            
            input_events = match.format_events_for_syncer()
            raw_tracking, input_tracking = match.format_tracking_for_syncer()

            # BePro data has many dead frames, so only events with nearby alive tracking data are used.
            if "bepro" in root_dir.lower():
                input_events = match.get_alive_events(input_events, input_tracking)

            syncer = ELASTIC(input_events, input_tracking)
            syncer.run()
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            return
        
        # save synchronized data
        match.lineup.to_parquet(os.path.join(elastic_dir, match_id, "meta_data.parquet"))
        syncer.events.to_parquet(os.path.join(elastic_dir, match_id, "event.parquet"))
        input_tracking.to_parquet(os.path.join(elastic_dir, match_id, "tracking.parquet"))
        raw_tracking.to_parquet(os.path.join(elastic_dir, match_id, "raw_tracking.parquet"))
    
    print(f"Available CPU cores: {os.cpu_count()}")
    with tqdm_joblib(tqdm(desc="Processing matches", total=len(match_id_list))):
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_single_elastic_wrapper)((match_id, root_dir, elastic_dir))
            for match_id in match_id_list
        )

def single_processing_elastic(match_id_list, root_dir, elastic_dir):
    for match_id in tqdm(match_id_list):
        os.makedirs(os.path.join(elastic_dir, match_id), exist_ok=True)
        if os.path.exists(os.path.join(elastic_dir, match_id, "tracking.parquet")):
            print(f"Match {match_id} already processed. Skipping.")
            continue
        
        try:
            if "bepro" in root_dir.lower():
                match = BeproData(root_dir, match_id, load_tracking=True)
            elif "dfl" in root_dir.lower():
                match = SportecData(root_dir, match_id, load_tracking=True)
            else:
                raise ValueError("Unknown data source. Please check the root_dir.")

            input_events = match.format_events_for_syncer()
            raw_tracking, input_tracking = match.format_tracking_for_syncer()

            # BePro data has many dead frames, so only events with nearby alive tracking data are used.
            if "bepro" in root_dir.lower():
                input_events = match.get_alive_events(input_events, input_tracking)
            
            syncer = ELASTIC(raw_tracking, input_tracking)
            syncer.run()
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            continue

        # save synchronized data
        match.lineup.to_parquet(os.path.join(elastic_dir, match_id, "meta_data.parquet"))
        syncer.events.to_parquet(os.path.join(elastic_dir, match_id, "event.parquet"))
        input_tracking.to_parquet(os.path.join(elastic_dir, match_id, "tracking.parquet"))  
        raw_tracking.to_parquet(os.path.join(elastic_dir, match_id, "raw_tracking.parquet"))
        
if __name__ == "__main__":
    """
        python elastic/convert_elastic.py \
        --data_dir ~/geonhee/Data/dfl/raw \
        --save_dir ./data/dfl/elastic \
        --n_jobs -1

       python elastic/convert_elastic.py \
        --data_dir ~/geonhee/Data/bepro/raw_research/2024 \
        --save_dir ./data/bepro/elastic \
        --n_jobs -1
    """
    
    parser = argparse.ArgumentParser(description="Convert DFL data to ELASTIC format using multi-processing")
    parser.add_argument('--data_dir', type=str, default="./data/dfl/raw",
                        help='Root directory containing raw DFL data')
    parser.add_argument('--save_dir', type=str, default="./data/dfl/elastic",
                        help='Output directory for ELASTIC formatted data.')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for processing (default: 1)')
    args = parser.parse_args()
    
    
    # It can take about one or two minutes to parse tracking data using kloppy
    os.makedirs(args.save_dir, exist_ok=True)

    match_id_list = os.listdir(args.data_dir)
    print(f"Game IDs: {match_id_list}")

    if args.n_jobs != 1:
        multi_processing_elastic(match_id_list, args.data_dir, args.save_dir, n_jobs=args.n_jobs)
    else:
        single_processing_elastic(match_id_list, args.data_dir, args.save_dir)

    print("Done")