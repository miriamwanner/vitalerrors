import os

def main():



    datasets = ["hotpotqa", "naturalquestions", "triviaqa", "bright", "wildhallucinations", "factscore"]
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
        "wildhallucinations": ["cult_ent_1", "cult_ent_2", "cult_ent_3", "cult_ent_4", "geographic"],
        "factscore": ["bios"],
        "hotpotqa": ["val_1", "val_2"],
        "naturalquestions": ["val_1", "val_2"],
        "triviaqa": ["rc_1", "rc_2"]
    }
    prompts = ["normal", "missing", "wrong"]

    for d in datasets:
        for s in subsets[d]:
            for p in prompts:
                dir_path = "data/" + d + "/" + s + "/" + p + "/"
                # print(dir_path)
                for file in os.listdir(dir_path):
                    print(file)
                    if file != "data.jsonl":
                        os.remove(dir_path + file)




if __name__ == "__main__":
    main()