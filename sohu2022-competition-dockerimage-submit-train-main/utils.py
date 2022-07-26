from collections import Counter
import numpy as np

def merge_idx(idxArr, span, content):
    assert len(idxArr) >= 1
    if len(idxArr)==1:
        return content[max(0,idxArr[0]-span) : min(len(content),idxArr[0]+span)]
    i = 0
    ret = []
    while True:
        if i>=len(idxArr):break
        temp_i = i
        for j in range(i+1,len(idxArr)):
            if idxArr[j]-idxArr[temp_i] > 2*span:
                temp_i = j-1
                break
            else:
                temp_i = j
        ret.append(content[max(0,idxArr[i]-span) : min(len(content),idxArr[temp_i]+span)])    
        i = temp_i+1
    return '#'.join(ret)
            
def sample_context_by_list(entitys:list, content:str, length:int):
    '''
    通过entity列表筛选content中对应每个实体位置的前后文
    '''
    cnt = 0
    for entity in entitys:
        cnt += content.count(entity)
    if cnt == 0 or len(content)<=length:
        return content
    span = int(length/cnt/2)
    idxArr = []
    for entity in entitys:
        idx = content.find(entity,0)
        while idx != -1:
            idxArr.append(idx)
            idx = content.find(entity,idx+1)
    idxArr = sorted(idxArr)
    result = merge_idx(idxArr, span, content)
    return result


class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.75):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list
    
def split_sentence(content:str):
    for each in '。；！!?？':
        content = content.replace(each, each+'##')
    return content.split('##')

def bm25_sample(content, query, augment=1, length=512):
    """
    bm25相似度打分，然后进行截断，使其<=length
    :param query:
    :param content:
    :param length:
    :return:
    """

    if len(content) <= length:
        return [content]
    else:
        document_list = split_sentence(content.strip())
        rest_document_list = list()
        for document in document_list:
            if len(document) != 0:
                rest_document_list.append(document)

        document_list = rest_document_list
        # print(document_list)
        model = BM25_Model(document_list)
        scores = model.get_documents_score(query)
        index_scores = []
        for index, score_i in enumerate(scores):
#             print(index, document_list[index], score_i)
            index_scores.append((index, score_i))

        index_scores.sort(key=lambda index_score: index_score[1], reverse=True)
        
        save_document = [0] * len(document_list)
        content_length = 0
        
        for item in index_scores:
            index = item[0]
            save_document[index] = 1
            if content_length + len(document_list[index]) > length:
                break
            else:
                content_length += len(document_list[index])
        if augment ==1: # 不进行数据增强
            new_content = ""
            for i, save in enumerate(save_document):
                if save != 0:
                    new_content += document_list[i]
#             print(len(new_content),'|',new_content)
            return [new_content]
        # else: # 随机打乱句子进行数据增强
        #     if len(document_list) <=3: # 如果document句子数太短
        #         augment = min(len(document_list), augment) 
        #     new_content_arr = []
        #     new_content_index = []
        #     for i, save in enumerate(save_document):
        #         if save != 0:
        #             new_content_index.append(i)
        #     new_content_arr.append(''.join([document_list[n] for n in new_content_index]))
        #     for i in range(augment-1):
        #         random.shuffle(new_content_index)
        #         new_content_arr.append(''.join([document_list[n] for n in new_content_index]))
            
        #     # for i in range(augment):
        #     #     print(f'augment {i} len = {len(new_content_arr[i])} | {new_content_arr[i]}')
        #     return new_content_arr