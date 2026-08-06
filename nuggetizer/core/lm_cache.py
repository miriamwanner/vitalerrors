import time
import os
import pickle
import logging
import sys


class LMCache(object):
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache_dict = self.load_cache()
        self.add_n = 0

    def load_cache(self):
        if os.path.exists(self.cache_path):
            while True:
                try:
                    with open(self.cache_path, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    print("Pickle error: retry in 5sec...")
                    time.sleep(5)
        else:
            cache = {}
        return cache
    
    def save_cache(self):
        if self.add_n == 0:
            return
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache_dict, f)