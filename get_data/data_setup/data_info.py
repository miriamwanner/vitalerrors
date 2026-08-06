datasets = ["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"]
subsets = {
    "bright": [
        "biology",
        "earth_science",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
    ],
    "factscore": ["bios"],
    "wildhallucinations": [
        "cult_ent_1",
        "cult_ent_2",
        "cult_ent_3",
        "cult_ent_4",
        "geographic",
    ],
    "hotpotqa": ["val_1", "val_2"],
    "naturalquestions": ["val_1", "val_2"],
    "triviaqa": ["rc_1", "rc_2"],
}

gen_type = ["normal", "missing", "wrong"]
