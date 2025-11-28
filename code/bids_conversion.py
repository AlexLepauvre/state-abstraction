import pandas as pd
import json
from pathlib import Path
import os

bids_root = r"C:\Users\alexander.lepauvre\Documents\GitHub\state-abstraction\data\bids\limited_energy"
transitions_costs = {
    0: [1, 1],
    1: [2, 1],
    2: [1, 2],
    3: [2, 2]
}

events_json = {
    "trial_type": {
        "LongName": "Reward offer of the trial",
        "Description": "Reward level presented to the participant during the trial",
        "Levels": {f"offer={o}": "Participant earns {o} points if accept and sufficient energy" for o in [1, 2, 3, 4]}
    },
    "event_type": {
        "LongName": "Type of event within a trial",
        "Description": "Each trial consists of four distinct events",
        "Levels": {
            "response": "Participants are presented with an offer and decide on their response. Over as soon as participant press response button",
            "selection": "Highlight for 1 sec the response selected by the participant",
            "feedback": "Shows the consequence of the selected response to the participant (points increase and energy decrease, energy increase or nothing happens)",
            "iti": "Inter-trial interval",
        },
    },
    "response": {
        "LongName": "Response given by the participant",
        "Description": "Participant can accept or reject the offer",
        "Levels": {
            "0": "Accept",
            "1": "Reject",
        }
    },
    "points": {
        "LongName": "Points",
        "Description": "Points collected by the participants throughout the experiment",
        "units":"a.u."
    },
    "offer": {
        "LongName": "reward offer",
        "Description": "Reward offered to the participant if they accept and have sufficient energy, added to their points",
        "Levels": {
            "1": "Gain one point if accept",
            "2": "Gain two points if accept",
            "3": "Gain three point if accept",
            "4": "Gain four points if accept"
        }
    },
    "energy": {
        "LongName": "energy",
        "Description": "Energy available to the participant to accept offers",
        "Levels": {
            "0-6": "0 being no energy, 6 max energy"
        }
    },
    "current_cost": {
        "LongName": "Current cost",
        "Description": "Cost of the current 4 trials segments",
        "Levels": {
            "1": "Low cost",
            "2": "High cost",
        }
    },
    "future_cost": {
        "LongName": "Future cost",
        "Description": "Cost of the future 4 trials segments",
        "Levels": {
            "1": "Low cost",
            "2": "High cost",
        }
    },
    "t": {
        "LongName": "Time in current segment",
        "Description": "Time in current segment",
        "Levels": {
            "1-4": "Time in current segment"
        }
    },
    "StimulusPresentation": {
        "OperatingSystem": "n/a",
        "SoftwareName": "Psychtoolbox",
        "SoftwareRRID": "n/a",
        "SoftwareVersion": "n/a",
        "Code": "https://doi.org/10.5281/zenodo.5112965",
        "ScreenDistance": "n/a",
        "ScreenRefreshRate": 60,
        "ScreenResolution": "n/a",
        "ScreenSize": "n/a",
        "HeadStabilization": "none"
    }
}

dataset_description = {
  "Name": "Limited energy task",
  "BIDSVersion": "1.6.0",
  "DatasetType": "raw",
  "License": "CC0",
  "Authors": [
    "Florian Ott",
    "Eric Legler",
    "Stephan Kiebel"
  ],
  "Keywords": [
    "neuroscience",
    "decision making",
    "computational neuroscience"
  ],
  "Acknowledgements": "",
  "HowToAcknowledge": "Please cite this paper: https://www.sciencedirect.com/science/article/pii/S1053811922003469#sec0002",
  "Funding": [
    "DFG, Deutsche Forschungsgemeinschaft SFB 940 - Project number 178833530",
    "DFG, Deutsche Forschungsgemeinschaft TRR 265 - Project number 402170461",
    "Germany's Excellence Strategy - EXC 2050/1 - Project number 390696704 - Cluster of Excellence “Centre for Tactile Internet with Human-in-the-Loop” (CeTI) of Technische Universität Dresden",
  ],
  "EthicsApprovals": [
    "Institutional Review Board of the Technische Universität Dresden"
  ],
  "ReferencesAndLinks": [
    "https://www.sciencedirect.com/science/article/pii/S1053811922003469#sec0002",
    "Ott, F., Legler, E., & Kiebel, S. J. (2022). Forward planning driven by context-dependant conflict processing in anterior cingulate cortex. NeuroImage, 256, 119222."
  ],
  "DatasetDOI": "",
  "HEDVersion": "",
}

participants_json = {
    "species": {
        "Description": "species of the participant following ncbi taxonmys",
    },
    "age": {
        "Description": "age of the participant",
        "Units": "year"
    },
    "sex": {
        "Description": "sex of the participant as reported by the participant",
        "Levels": {
            "M": "male",
            "F": "female"
        }
    },
    "handedness": {
        "Description": "handedness of the participant as reported by the participant",
        "Levels": {
            "left": "left",
            "right": "right"
        }
    },
    "group": {
        "Description": "experimental group the participant belonged to",
        "Levels": {
            "read": "participants who read an inspirational text before the experiment",
            "write": "participants who wrote an inspirational text before the experiment"
        }
    }
}

def raw2bids(file, task="limited_energy"):
    if not os.path.isdir(bids_root):
        os.makedirs(bids_root)
    with open(Path(bids_root, 'dataset_description.json'), 'w') as outfile:
        json.dump(dataset_description, outfile, indent=4)
    # Load the data:
    raw_df = pd.read_csv(file, sep=',')

    # Add a subject column:
    sub_id = 1
    for i, row in raw_df.iterrows():
        if i > 10 and row[0] == 0:
            sub_id += 1
        raw_df.at[i, 'subject'] = f'sub-{sub_id}'

    subjects = list(raw_df["subject"].unique())
    participants_tsv = pd.DataFrame({
        'participant_id': subjects,
        'species': ['homo sapiens'] * len(subjects),
        'age': ['n/a'] * len(subjects),
        'sex': ['n/a'] * len(subjects),
        'handedness': ['n/a'] * len(subjects),
    }).reset_index(drop=True)
    participants_tsv.to_csv(Path(bids_root, 'participants.tsv'), sep="\t")
    with open(Path(bids_root, 'participants.json'), 'w') as outfile:
        json.dump(participants_json, outfile, indent=4)

    # Loop through each subject:
    for sub in raw_df['subject'].unique():
        # Extract the data of this subject:
        sub_data = raw_df[raw_df["subject"] == sub].reset_index(drop=True)
        events_df = pd.DataFrame()
        for trial_i, trial in sub_data.iterrows():
            # There are four events in a trial:
            events = ["response", "selection", "feedback", "iti"]
            events_onsets = [
                trial["response_onset"], 
                trial["selection_onset"], 
                trial["feedback_onset"], 
                trial["iti_onset"]
                ]
            events_duration = [
                trial["selection_onset"] - trial["response_onset"],
                trial["feedback_onset"] - trial["selection_onset"],
                trial["iti_onset"] - trial["feedback_onset"],
                None
            ]
            # Add to the table:
            events_df = pd.concat([events_df, pd.DataFrame({
                "trial_type": [f"offer={trial["reward"]}"] * 4,
                "event_type": events,
                "onset": events_onsets,
                "duration": events_duration,
                "response_time": [trial["reaction_time"]] * 4,
                "response":  [trial["response"]] * 4,
                "points": [trial["points"]] * 4,
                "offer": [trial["reward"]] * 4,
                "energy": [trial["energy"]] * 4,
                "current_cost": [trial["energy_cost"]] * 4,
                "future_cost": [transitions_costs[trial["transition"]][1]] * 4,
                "t": [trial["trial"] + 1] * 4,
            })]).reset_index(drop=True)
        
        # Save the subject's data:
        folder = Path(bids_root, sub, "beh")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        file = f'{sub}_task-{task}_events.tsv'
        events_df.to_csv(Path(folder, file), sep="\t")
        with open(Path(folder, f'{sub}_task-{task}_events.json'), 'w') as outfile:
            json.dump(events_json, outfile, indent=4)

raw2bids(r'.\data\raw_data\limited_energy\all_participants_data.csv', task="limited_energy")