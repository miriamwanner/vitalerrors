import json


class RankedSubclaims:
    def __init__(self, line):
        self.id2subclaim = self.get_id2subclaim(line["decomposition"])
        self.id2importance = self.get_id2importance(line["decomposition"])
        self.original_order = self.get_original(line["decomposition"])
        self.ranked_order = line["subclaim-importance-order"]

    def get_id2subclaim(self, decomposition):
        id2subclaim = {}
        for sentence in decomposition:
            for subclaim in sentence["decomp"]:
                id2subclaim[subclaim["id"]] = subclaim["text"]
        return id2subclaim

    def get_id2importance(self, decomposition):
        id2importance = {}
        for sentence in decomposition:
            for subclaim in sentence["decomp"]:
                id2importance[subclaim["id"]] = subclaim["importance"]
        return id2importance

    def get_original(self, decomposition):
        original_order = []
        for sentence in decomposition:
            for subclaim in sentence["decomp"]:
                original_order.append(subclaim["id"])
        return original_order

    def print_example(self, ranked=False):
        
        if ranked:
            iter_through = self.ranked_order
        else: 
            iter_through = self.original_order

        for i in iter_through:
            print("& " + i[1:] + " & " + self.id2importance[i] + " & " + self.id2subclaim[i] + "\\\\")


def main():
    root_dir = "data/naturalquestions/val_1"
    normal_path = root_dir + "/normal/new-metric-out.jsonl"
    missing_path = root_dir + "/missing/new-metric-out.jsonl"
    wrong_path = root_dir + "/wrong/new-metric-out.jsonl"

    count = 1
    with open(normal_path, 'r') as n, open(missing_path, 'r') as m, open(wrong_path, 'r') as w:
        for n_line, m_line, w_line in zip(n, m, w):
            n_line, m_line, w_line = json.loads(n_line), json.loads(m_line), json.loads(w_line)
            if count == 23:
                normal = RankedSubclaims(n_line)
                missing = RankedSubclaims(m_line)
                wrong = RankedSubclaims(w_line)

                print("======================= NORMAL =======================")
                normal.print_example()
                print()
                normal.print_example(ranked=True)
                print()
                print("======================= MISSING =======================")
                missing.print_example()
                print()
                missing.print_example(ranked=True)
                print()
                print("======================= WRONG =======================")
                wrong.print_example()
                print()
                wrong.print_example(ranked=True)

                break
            count += 1

if __name__ == "__main__":
    main()