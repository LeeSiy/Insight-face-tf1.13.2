from multiprocessing import Lock
from pathos.pools import ThreadPool as pp
import mxnet as mx
import pickle
import glob

class JoinRecIO(object):
    def __init__(self,prefix_read='path/to/merge/recs/idxs'):
        
        
        flname_recs = sorted(glob.glob(prefix_read + '*/*.rec'))
        flname_idxs = sorted(glob.glob(prefix_read + '*/*.idx'))
        
        self.nameTuples = list(zip(flname_idxs, flname_recs))
        
        self.record_target = mx.recordio.MXIndexedRecordIO(idx_path=prefix_read+'/Train.idx', uri=prefix_read +'/Train.rec', flag='w')
        
        self.lock = Lock()
        self.global_idx = 0
        
        
    def thread_write(self,nthreads=6):
        
        pool = pp(nthreads)
        pool.map(self.write_rec,self.nameTuples)
            
        self.record_target.close()
    
    def write_rec(self, names_idx_rec):
        name1, name2 = names_idx_rec
        record_read = mx.recordio.MXIndexedRecordIO(idx_path=name1, uri=name2, flag='r')

        for key in record_read.keys:
            read_in = record_read.read_idx(key)
            self.lock.acquire()
            self.record_target.write_idx(self.global_idx,read_in)
            self.global_idx += 1
            self.lock.release()








myJoin = JoinRecIO()
myJoin.thread_write()
