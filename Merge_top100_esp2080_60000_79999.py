# /share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer $ python3 Merge_top100.py

# /share/nas165/chunliang/voxceleb_trainer/write_empty_txt.py
class score_merge():
    def __init__(self):

        self.start = 60000 # 這裡開始
        self.end = 79999  # 這裡結束


        self.n_top = 100
        self.score_dir = '/home/chengsam/esp2080_test_score_6w_to_8w/'
        # self.score_dir = '/share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer/test_score/'
        # self.merge_path = '/share/nas165/chengsam/voxceleb2020_prog/fully_supervised_speaker_verification/voxceleb_trainer/top_100_merge/'
        self.merge_path = '/home/chengsam/esp2080_top_100_merge_60000_79999/'
        # 000000.wav to 118438.wav

    def preview_top100(self, idx):
        # with open(self.merge_path+'score_sorted_'+idx+'.txt', 'w')
        preview_list = []
        with open(self.merge_path+'score_sorted_'+str(idx)+'.txt') as lines:
            while True:
                line = lines.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;
                data = line.split();
                preview_list.append([float(data[0]), data[1]])
        # print(sorted(preview_list, key=lambda x:x[0], reverse=False))
        return preview_list

    def write_preview_file(self, idx, pre_list):
        with open(self.merge_path+'score_sorted_'+str(idx)+'.txt', 'w') as out: # 處理現在的idx(善未考慮之前的)
            for doc in pre_list:
                out.write("%s %s\n"%(doc[0], doc[1]))

    def process(self):
        for idx in range( self.start, self.end+1 ): # 檔案的跳動
            print('score_sorted_'+str(idx)+'.txt')
            ref_list = []
            com_list = {}
            with open(self.score_dir+'test_score_'+str(idx)+'.txt') as lines:
                while True:
                    line = lines.readline();
                    if (not line): #  or (len(all_scores)==1000) 
                        break;
                    data = line.split();
                    ref_name = data[1]
                    com_name = data[2]
                    data[0] = float(data[0])
                    ref_idx = int(data[1].replace('.wav',''))
                    com_idx = int(data[2].replace('.wav',''))
                    if ref_idx == idx: # 自己的
                        ref_list.append([data[0], com_name])

                        preview_list = self.preview_top100(com_idx)
                        if len(preview_list) > 99:
                            if data[0]> float(preview_list[-1][0]): # 如果新的資料比舊的大，加入list
                                preview_list.append([data[0], ref_name])
                                new_preview_list = sorted(preview_list, key=lambda x:x[0], reverse=True)
                                self.write_preview_file(com_idx, new_preview_list[:self.n_top]) # 可能出現長度問題
                        else:
                            preview_list.append([data[0], ref_name])
                            new_preview_list = sorted(preview_list, key=lambda x:x[0], reverse=True)
                            self.write_preview_file(com_idx, new_preview_list[:self.n_top]) # 可能出現長度問題


            preview_list = self.preview_top100(idx)
            # print(preview_list)
            ref_list.extend(preview_list)
            sort_list = sorted(ref_list, key=lambda x:x[0], reverse=True) # 由大到小
            with open(self.merge_path+'score_sorted_'+str(idx)+'.txt', 'w') as out: # 處理現在的idx(善未考慮之前的)
                for doc in sort_list[:self.n_top]:
                    out.write("%s %s\n"%(doc[0], doc[1])) # 分數 檔名

if __name__ == '__main__': 
    sm = score_merge()
    sm.process()

