

class RSSampler: # Regulation Signal Sampler
    rs_file_data = list()
    def __init__(
        self,
        rs_file_path: str
    ):
        self.rs_file_path = rs_file_path
        self.rs_len = 0
        self.read_rs_file_data()
    def read_rs_file_data(self):
        f = open(self.rs_file_path, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            self.rs_len += 1
            self.rs_file_data.append(float(line))
        print(f"[RSSampler]: read regulation signal file ({self.rs_file_path}) len({self.rs_len})")

    def sample(self, new_len: int = 450, diff: float = 0.5): 
        """
        samples the regulation signal file
        to reduce it by a given ratio by averaging.

        Args:
            new_len_sec (int): rs_len_sec//20 <= new_len_sec <= rs_len_sec
            reduces the original regulation signal file by averaging.
            max reduction ratio is 20 which means a 3600 sec file can be reduced 
            down to 3600//20=180 sec file.
        """
        if new_len > self.rs_len:
            return [None, False]
        ratio = self.rs_len//new_len
        if ratio > 20:
            return [None, False]
        
        last_ind = int(new_len*ratio)
        print(f"[RSSampler]: Truncating {self.rs_len-last_ind} samples from original rs file", flush=True)
        print(f"[RSSampler]: sampling ratio: {ratio}", flush=True)
        truncated = self.rs_file_data[:last_ind] if last_ind > 0 else self.rs_file_data
        sampled = list()
        for start in range(new_len):
            sample = truncated[start*ratio:(start+1)*ratio]
            sampled.append(sum(sample)/len(sample))
        
        
        for ind in range(0, len(sampled)-1):
            if abs(sampled[ind]-sampled[ind+1]) > diff:
                print(f"{sampled[ind]} and {sampled[ind+1]}")
                return [sampled, False]
        
        return [sampled, True]
    def get_chunk(self, num_chunks: int, chunk_ind: int):
        if chunk_ind >= num_chunks:
            print(f"[RSSampler/chunker]: chunk index {chunk_ind} has to be in range [0,{num_chunks})", flush=True)
            return [None, False]
        if self.rs_len % num_chunks is not 0:
            print(f"[RSSampler/chunker]: RS signal length ({self.rs_len}) not divisible number of requested chunks ({chunk_ind})", flush=True)
            return [None, False]
        
        chunk = self.rs_file_data[chunk_ind*(self.rs_len//num_chunks):(chunk_ind+1)*(self.rs_len//num_chunks)]
        return [chunk, True]

def test_sample():
    file_path = "/home/ajaha004/repos/ROCm-docker-with-controller/scripts/frequency_regulator_power_cap/data/reg_sig_highreg"
    sampler = RSSampler(rs_file_path=file_path)
    
    print(sampler.sample(1800)[1])
    print(sampler.sample(900)[1])
    print(sampler.sample(450)[1])
    print(sampler.sample(225)[1])
    
    
    i = 0
    num_chunks = 30
    all_c =  sampler.get_chunk(1, 0)[0]
    for k in range(num_chunks):
        chunk = sampler.get_chunk(num_chunks, k)
        assert chunk[1], f"orignal rs file length ({len(all_c)}) is not divisable by num chunks requested ({num_chunks})"
        c1 = chunk[0]
        for j in range(len(c1)):
            assert all_c[i] == c1[j], f"values differ (original,chunk_ind,ind) @{i}({all_c[i]}),{k},{j}({c1[j]})"
            i +=1

if __name__ == "__main__":
    test_sample()