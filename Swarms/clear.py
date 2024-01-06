
import pickle
import shelve

with shelve.open('cache.db', flag='c', protocol=pickle.HIGHEST_PROTOCOL, writeback=False) as db:
    for key in ():
        del db[key]
